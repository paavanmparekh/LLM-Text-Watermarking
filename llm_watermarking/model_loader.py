"""
model_loader.py — Load Mistral-7B-Instruct with 4-bit BitsAndBytes quantization.

Usage
-----
    from llm_watermarking.model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer()
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import Config, config as default_config


def load_model_and_tokenizer(cfg: Config = None):
    """
    Load the causal LM and tokenizer specified in *cfg*.

    Parameters
    ----------
    cfg : Config, optional
        Configuration object. Defaults to the module-level singleton.

    Returns
    -------
    model : transformers.PreTrainedModel
    tokenizer : transformers.PreTrainedTokenizer
    """
    if cfg is None:
        cfg = default_config

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16)

    bnb_config = None
    if cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    print(f"Loading {cfg.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token   # required for batch padding

    model_kwargs = {
        "device_map": cfg.device_map,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        **model_kwargs
    )
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer
