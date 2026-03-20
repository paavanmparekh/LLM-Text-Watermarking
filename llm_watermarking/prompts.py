"""
prompts.py — Prompt management for the LLM Watermarking project.

Provides a PromptLoader class that can be initialised with a custom
list of prompts or falls back to a set of diverse default prompts.
Prompts follow the Mistral [INST] ... [/INST] chat template.

Usage
-----
    from llm_watermarking.prompts import PromptLoader

    loader = PromptLoader()                   # default prompts
    loader = PromptLoader(prompts=[...])      # custom prompts
    loader = PromptLoader.from_file("p.txt") # one prompt per line

    for prompt in loader:
        print(prompt)
"""

from typing import List, Optional


DEFAULT_PROMPTS: List[str] = [
    "[INST] Explain the concept of zero-knowledge proofs. [/INST]",
    "[INST] Write a short summary about the history of the Roman Empire. [/INST]",
    "[INST] What are the main differences between classical computing and quantum computing? [/INST]",
    "[INST] Describe a futuristic city in the year 2050 using vivid imagery. [/INST]",
    "[INST] Provide a step-by-step guide to baking a chocolate cake. [/INST]",
]


class PromptLoader:
    """
    Manages the collection of prompts used during generation.

    Parameters
    ----------
    prompts : list of str, optional
        Custom prompt strings. If *None*, DEFAULT_PROMPTS are used.
    """

    def __init__(self, prompts: Optional[List[str]] = None) -> None:
        self.prompts: List[str] = prompts if prompts else list(DEFAULT_PROMPTS)

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath: str) -> "PromptLoader":
        """Load prompts from a plain-text file (one prompt per line)."""
        with open(filepath, "r", encoding="utf-8") as fh:
            prompts = [line.strip() for line in fh if line.strip()]
        return cls(prompts=prompts)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def get_prompts(self) -> List[str]:
        """Return the full list of prompts."""
        return self.prompts

    def add_prompt(self, prompt: str) -> None:
        """Append a new prompt at runtime."""
        self.prompts.append(prompt)

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts)

    def __repr__(self) -> str:  # pragma: no cover
        return f"PromptLoader({len(self.prompts)} prompts)"
