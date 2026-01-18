# -*- coding: utf-8 -*-
"""
GPT Model implementation for language modeling.

A modified GPT architecture with value embeddings, skip connections,
and document-aware causal masking.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from .components import CastedLinear, Block
from auon.utils import norm, next_multiple_of_n


class GPT(nn.Module):
    """
    GPT language model with value embeddings and skip connections.
    
    This implementation includes:
    - Token and value embeddings
    - Skip connections between encoder/decoder layers
    - Document-aware causal masking
    - FP8-enabled output projection
    
    Args:
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        model_dim: Model dimension
        max_seq_len: Maximum sequence length
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        
        # Round vocab size to multiple of 128 for efficiency
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, model_dim)
        
        # Value embeddings (3 copies for different layers)
        self.value_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, model_dim) for _ in range(3)
        ])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, max_seq_len, i)
            for i in range(num_layers)
        ])
        
        # Output projection with FP8
        self.lm_head = CastedLinear(
            model_dim, vocab_size,
            use_fp8=True,
            x_s=(model_dim ** 0.5) / 448,
            w_s=24 / 448,
            grad_s=1 / 448,
        )
        self.lm_head.weight.detach().zero_()
        
        # Learnable scalars for skip connections and attention mixing
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
        ]))
        
        # Learning rate multipliers
        for p in self.embed.parameters():
            p.lr_mul = 75.0
        for p in self.value_embeds.parameters():
            p.lr_mul = 75.0
        self.lm_head.weight.lr_mul = 27.5
        self.scalars.lr_mul = 5.0

    def create_blockmasks(
        self,
        input_seq: Tensor,
        sliding_window_num_blocks: Tensor,
    ) -> tuple[BlockMask, BlockMask]:
        """
        Create block masks for document-aware causal attention.
        
        Args:
            input_seq: Input sequence tensor
            sliding_window_num_blocks: Number of blocks for sliding window
            
        Returns:
            Tuple of (long_block_mask, short_block_mask)
        """
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(
                dim=-1, descending=False, stable=True
            ).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_any = block_idx[:, None] >= block_idx
        causal_all = block_idx[:, None] > block_idx
        
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        
        doc_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        doc_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        
        blockmask_any = causal_any & doc_any
        blockmask_all = causal_all & doc_all
        
        partial_kv_num, partial_kv_idx = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num, full_kv_idx = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(
                    partial_kv_num,
                    torch.clamp_min(window_size_blocks - full_kv_num, 1),
                ),
                partial_kv_idx,
                torch.clamp_max(full_kv_num, window_size_blocks - 1),
                full_kv_idx,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor,
        sliding_window_num_blocks: Tensor,
    ) -> Tensor:
        """
        Forward pass computing language modeling loss.
        
        Args:
            input_seq: Input token sequence (1D tensor)
            target_seq: Target token sequence
            sliding_window_num_blocks: Number of blocks for sliding window attention
            
        Returns:
            Cross-entropy loss
        """
        assert input_seq.ndim == 1
        
        # Compute value embeddings
        ve = [vemb(input_seq) for vemb in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        # Create block masks
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [
            long_bm, short_bm, short_bm, short_bm,
            long_bm, short_bm, short_bm, long_bm,
            short_bm, short_bm, short_bm, long_bm,
        ]
        assert len(block_masks) == len(self.blocks)

        # Embed input
        x = x0 = norm(self.embed(input_seq)[None])

        # Extract scalars
        skip_weights = self.scalars[:len(self.blocks) // 2]
        lambdas = self.scalars[1 * len(self.blocks):3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks):5 * len(self.blocks)].view(-1, 2)
        n = len(self.blocks) // 2

        # Forward through blocks with skip connections
        skip_connections = []
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
            if i < n:
                skip_connections.append(x)

        # Output projection
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1) ** 0.5))
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_seq,
            reduction="sum" if self.training else "mean",
        )
        
        return loss
