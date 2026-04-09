import random
from typing import List

class TextModifier:
    """
    Utility for applying adversarial or natural modifications to text
    to evaluate watermark robustness.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        # A small sample of common words for insertions
        self.filler_words = [
            "actually", "basically", "so", "then", "just", "very", "really",
            "maybe", "perhaps", "actually", "literally", "honestly"
        ]

    def apply_substitutions(self, text: str, p: float) -> str:
        """
        Randomly replaces words with probability p.
        Replaces each selected word with a random filler word.
        """
        if p <= 0: return text
        words = text.split()
        new_words = []
        for word in words:
            if self.rng.random() < p:
                new_words.append(self.rng.choice(self.filler_words))
            else:
                new_words.append(word)
        return " ".join(new_words)

    def apply_insertions(self, text: str, p: float) -> str:
        """
        Randomly inserts filler words between existing words with probability p.
        """
        if p <= 0: return text
        words = text.split()
        new_words = []
        for word in words:
            new_words.append(word)
            if self.rng.random() < p:
                new_words.append(self.rng.choice(self.filler_words))
        return " ".join(new_words)

    def apply_deletions(self, text: str, p: float) -> str:
        """
        Randomly deletes words with probability p.
        """
        if p <= 0: return text
        words = text.split()
        new_words = [w for w in words if self.rng.random() > p]
        return " ".join(new_words) if new_words else ""
