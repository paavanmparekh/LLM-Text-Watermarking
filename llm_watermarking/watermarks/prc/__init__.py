"""
PRC (Pseudorandom Error-Correcting Code) primitive.

This module implements the paper's zero-bit public-key LDPC-PRC0 construction
(Construction 2) from:
  Miranda Christ, Sam Gunn — "Pseudorandom Error-Correcting Codes"
  ePrint 2024/235, arXiv:2402.09370

Only the PRC primitive lives here (KeyGen / Encode / Decode). The actual LLM
watermarking scheme (Setup/Wat/Detect; Figure 3 / Section 7) is implemented in
later project phases.
"""

from .prc import (
    LDPCPRC0,
    LDPCPRC0Params,
    LDPCPRC0PublicKey,
    LDPCPRC0SecretKey,
)
from .generation import PRCWatermark
from .detection import PRCWatermarkDetector

__all__ = [
    "LDPCPRC0",
    "LDPCPRC0Params",
    "LDPCPRC0PublicKey",
    "LDPCPRC0SecretKey",
    "PRCWatermark",
    "PRCWatermarkDetector",
]
