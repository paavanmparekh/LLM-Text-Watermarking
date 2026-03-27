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

WATERMARK_REGISTRY: dict = {
    "Undetectable": UndetectableWatermark,
    # "PRC": PRCWatermark,       # add in future phases
    # "KGW": KGWWatermark,       # add in future phases
}

__all__ = ["WATERMARK_REGISTRY", "UndetectableWatermark"]
