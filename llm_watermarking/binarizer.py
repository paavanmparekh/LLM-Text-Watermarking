"""
binarizer.py — Vocabulary binarization utilities for binary-token watermarking schemes.

Decomposes token sampling over a vocabulary of size V into a sequence of
`blen = ceil(log2(V))` binary (0/1) decisions, so that watermarking schemes
(e.g. Undetectable, PRC) can operate over a binary token space {0, 1} while
the model itself remains unchanged.

Usage
-----
    from llm_watermarking.binarizer import build_binary_vocab, compute_bit_probs

    bit_length, tok2id, id2tok = build_binary_vocab(tokenizer)
    p0, p1 = compute_bit_probs(probs, bit_index=0, bit_length=bit_length, prefix=0)
"""

import math

import torch


def build_binary_vocab(tokenizer):
    """
    Pre-compute the binary decomposition metadata for a tokenizer vocabulary.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer

    Returns
    -------
    bit_length : int
        Number of bits needed to index any token: ceil(log2(vocab_size)).
    tok2id : dict[str, int]
        Token string → token ID mapping (from tokenizer).
    id2tok : dict[int, str]
        Token ID → token string mapping (inverse of tok2id).
    """
    bit_length = math.ceil(math.log2(len(tokenizer)))
    tok2id = tokenizer.get_vocab()
    id2tok = {v: k for k, v in tok2id.items()}
    return bit_length, tok2id, id2tok


def compute_bit_probs(
    probs: torch.Tensor,
    bit_index: int,
    bit_length: int,
    prefix: int,
) -> tuple:
    """
    Compute probability mass for bit=0 and bit=1 at position `bit_index`,
    conditioned on the bits already decided (encoded as `prefix`).

    The token IDs consistent with `prefix` at bits [0 .. bit_index-1] form a
    contiguous range [lo, hi).  Within that range we split by the value of bit
    `bit_index` (MSB-first) and sum the respective probability masses.

    Parameters
    ----------
    probs : torch.Tensor, shape [vocab_size]
        Full softmax probability distribution over the vocabulary (CPU tensor).
    bit_index : int
        Which bit we are deciding next (0 = most-significant bit).
    bit_length : int
        Total number of bits per token (= ceil(log2(vocab_size))).
    prefix : int
        The bit decisions made so far, packed as an integer
        (accumulated by left-shift + new bit at each step).

    Returns
    -------
    p0 : torch.Tensor (scalar)
        Total probability mass for tokens where bit `bit_index` is 0.
    p1 : torch.Tensor (scalar)
        Total probability mass for tokens where bit `bit_index` is 1.
    """
    # The prefix so far covers `bit_index` bits (MSBs).
    # Remaining bits after this decision: (bit_length - bit_index - 1)
    remaining_after = bit_length - bit_index - 1

    lo = prefix << (remaining_after + 1)           # first token ID for this prefix
    hi = min(lo + (1 << (remaining_after + 1)), len(probs))   # exclusive upper bound

    if lo >= hi:
        zero = torch.tensor(0.0)
        return zero, zero

    # Vectorised: check the value of bit at position `remaining_after` within [lo, hi)
    ids = torch.arange(lo, hi, dtype=torch.long)
    bit_vals = (ids >> remaining_after) & 1        # 0 or 1 for each token

    p_slice = probs[lo:hi]
    p0 = p_slice[bit_vals == 0].sum()
    p1 = p_slice[bit_vals == 1].sum()
    return p0, p1
