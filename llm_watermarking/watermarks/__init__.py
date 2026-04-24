"""
watermarks/__init__.py — Registry of available watermarking schemes.

Each scheme is a class with a `generate(model, tokenizer, prompt, ...)` method.
Binary-token schemes (Undetectable, PRC) activate the binarizer internally.
Vocabulary-level schemes (KGW, etc.) do not touch the binarizer.

Usage
-----
    from llm_watermarking.watermarks import WATERMARK_REGISTRY

    scheme_cls = WATERMARK_REGISTRY["Undetectable"]
    scheme = scheme_cls(cfg)
    result  = scheme.generate(model, tokenizer, prompt)
"""

from .undetectable.generation import UndetectableWatermark
from .prc.generation import PRCWatermark
from .prc import LDPCPRC0, LDPCPRC0Params, LDPCPRC0PublicKey, LDPCPRC0SecretKey

WATERMARK_REGISTRY: dict = {
    "Undetectable": UndetectableWatermark,
    "PRC": PRCWatermark,
    # "KGW": KGWWatermark,       # add in future phases
}

__all__ = [
    "WATERMARK_REGISTRY",
    "UndetectableWatermark",
    "PRCWatermark",
    "LDPCPRC0",
    "LDPCPRC0Params",
    "LDPCPRC0PublicKey",
    "LDPCPRC0SecretKey",
]
