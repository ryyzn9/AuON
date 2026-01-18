# -*- coding: utf-8 -*-
"""
AuON: Alternative Unit-norm momentum-updates by Normalized nonlinear scaling.

A linear-time optimizer that achieves remarkable performance without producing 
semi-orthogonal matrices while preserving structure to guide better-aligned 
progress and recondition ill-posed updates.
"""

from .optimizer import AuON, HybridAuON
from .muon import Muon
from .adam import Adam
from .utils import zeropower_via_newtonschulz5

__version__ = "0.1.0"
__all__ = ["AuON", "HybridAuON", "Muon", "Adam", "zeropower_via_newtonschulz5"]
