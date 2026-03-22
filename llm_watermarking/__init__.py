"""
LLM Watermarking — Modular Python Package
==========================================
A research codebase for implementing and evaluating LLM text watermarking schemes
using Mistral-7B-Instruct-v0.2 on GPU (e.g. Google Colab T4).

Modules
-------
config          : Model name, generation defaults, file paths
model_loader    : Load / cache the Mistral model with 4-bit quantization
prompts         : PromptLoader — define and manage input prompts
generation      : BaselineLogitTracker + LLMGenerator (hook-friendly)
evaluation      : Evaluator — perplexity, Distinct-N, entropy metrics
visualization   : All plotting utilities
pipeline        : run_pipeline() — end-to-end orchestrator
"""

from .config import Config
from .model_loader import load_model_and_tokenizer
from .prompts import PromptLoader
from .generation import BaselineLogitTracker, LLMGenerator
from .evaluation import Evaluator
from .visualization import plot_evaluation_metrics
from .pipeline import run_pipeline
from .binarizer import build_binary_vocab, compute_bit_probs
from .watermarks import WATERMARK_REGISTRY

__all__ = [
    "Config",
    "load_model_and_tokenizer",
    "PromptLoader",
    "BaselineLogitTracker",
    "LLMGenerator",
    "Evaluator",
    "plot_evaluation_metrics",
    "run_pipeline",
    "build_binary_vocab",
    "compute_bit_probs",
    "WATERMARK_REGISTRY",
]
