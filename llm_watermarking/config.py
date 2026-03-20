"""
config.py — Central configuration for the LLM Watermarking project.

All constants and tuneable hyper-parameters live here so that no
magic numbers are scattered across other modules.
"""

from dataclasses import dataclass, field
import os


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    #  Model                                                               #
    # ------------------------------------------------------------------ #
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    """HuggingFace model ID."""

    load_in_4bit: bool = True
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
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True

    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    output_dir: str = "outputs"
    """Directory where results and plots are saved."""

    results_file: str = "baseline_results.jsonl"
    """JSONL file for raw generation + evaluation results."""

    @property
    def results_path(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, self.results_file)


# Singleton — import and use directly.
config = Config()
