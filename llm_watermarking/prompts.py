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

import torch
from datasets import load_dataset


DEFAULT_PROMPTS: List[str] = [
    # LOW ENTROPY: Highly constrained, deterministic facts and structured lists
    "Instruct: List the first ten elements of the periodic table in exact order.\nOutput:",

    # MEDIUM ENTROPY: Standard Q&A, explanations, instructional (content is factual but phrasing varies)
    "Instruct: Provide a step-by-step guide to baking a basic chocolate cake.\nOutput:",

    # HIGH ENTROPY: Creative, abstract, open-ended brainstorming (infinite valid paths)
    "Instruct: Describe a chaotic, neon-lit cyberpunk market in the year 2099 using vivid, sensory imagery.\nOutput:",
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

    @classmethod
    def from_c4(cls, tokenizer, num_samples: int = 500) -> "PromptLoader":
        """
        Load prompts from the allenai/c4 (en) dataset.
        Takes the first 30 tokens of each sample's text as the prompt.
        """
        print(f"Loading {num_samples} samples from allenai/c4 dataset... (this may take a moment)")
        
        # We use the 'validation' split because it has 8 shards instead of 1024,
        # which loads almost instantly even with streaming=True.
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        
        prompts = []
        # Filter for texts that are long enough to actually have a decent prompt
        # We'll just tokenize the first chunks until we get num_samples
        for item in dataset:
            text = item["text"]
            # Rough fast filter: text must be at least ~150 chars to likely have 30 tokens
            if len(text) < 150:
                continue
                
            # Tokenize to exact first 30 tokens
            tokens = tokenizer(text, truncation=True, max_length=30, add_special_tokens=False)["input_ids"]
            
            if len(tokens) >= 30:
                # Decode back to string
                prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)
                prompts.append(prompt_text)
                
            if len(prompts) >= num_samples:
                break
                
        print(f"Successfully extracted {len(prompts)} prompts from C4.")
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
