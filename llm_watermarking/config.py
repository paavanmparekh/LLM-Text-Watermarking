"""
config.py — Central configuration for the LLM Watermarking project.

All constants and tuneable hyper-parameters live here so that no
magic numbers are scattered across other modules.
"""

from dataclasses import dataclass, field
import os
from typing import Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    #  Model                                                               #
    # ------------------------------------------------------------------ #
    model_name: str = "microsoft/phi-2"
    """HuggingFace model ID."""

    load_in_4bit: bool = False
    """Enable 4-bit QLoRA quantization via BitsAndBytes."""

    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"   # "float16" on older GPUs

    device_map: str = "auto"
    """Device placement strategy for model sharding."""

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #
    max_new_tokens: int = 150
    temperature: float = 1.0
    top_p: float = 0.95
    do_sample: bool = True

    # ------------------------------------------------------------------ #
    #  Paths & Watermark Params                                            #
    # ------------------------------------------------------------------ #
    output_dir: str = "outputs"
    """Directory where results and plots are saved."""

    watermark: Optional[str] = None
    """Active watermarking scheme name (e.g. 'Undetectable', 'PRC').
    None means standard baseline generation (no watermark)."""
    
    watermark_key: Optional[str] = None
    """Hex-encoded string of the secret key. If None, random key is used."""
    
    lambda_entropy: float = 10.0
    """Security parameter λ for Undetectable watermarking (bits)."""

    @property
    def results_file(self) -> str:
        """JSONL filename derived from the active watermarking scheme."""
        if self.watermark:
            return f"{self.watermark.lower()}_results.jsonl"
        return "baseline_results.jsonl"

    @property
    def results_path(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, self.results_file)


# Singleton — import and use directly.
config = Config()
