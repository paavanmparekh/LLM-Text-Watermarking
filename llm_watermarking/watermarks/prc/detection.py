import hashlib
import math
from typing import Any, Dict, List, Optional

from ...binarizer import build_binary_vocab
from .prc import LDPCPRC0, LDPCPRC0Params


def _seed_to_uint64(seed: bytes) -> int:
    digest = hashlib.sha256(seed).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _mask_from_seed(seed: bytes, *, block_idx_1based: int, n_bits: int) -> int:
    if block_idx_1based <= 0:
        raise ValueError("block_idx_1based must be >= 1")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    out = 0
    produced = 0
    counter = 0
    while produced < n_bits:
        h = hashlib.sha256()
        h.update(seed)
        h.update(b"PRC_MASK")
        h.update(block_idx_1based.to_bytes(4, "big", signed=False))
        h.update(counter.to_bytes(4, "big", signed=False))
        chunk = h.digest()
        for byte in chunk:
            for bit in range(8):
                if produced >= n_bits:
                    break
                if (byte >> bit) & 1:
                    out |= 1 << produced
                produced += 1
        counter += 1
    return out


def _pack_bits_lsb_first(bits: str) -> int:
    x = 0
    for i, ch in enumerate(bits):
        if ch == "1":
            x |= 1 << i
    return x


class PRCWatermarkDetector:
    """
    PRC watermark detector (Figure 3 / Algorithm 3), specialized to a
    length-preserving zero-bit PRC (LDPC-PRC0).

    For practicality we scan all length-n windows of the bitstring and all
    possible OTP masks a_ℓ that could have been used during generation.
    """

    def __init__(
        self,
        key: bytes,
        prc_params: LDPCPRC0Params,
        tokenizer: Any = None,
        *,
        robust_scan: bool = False,
    ) -> None:
        self.key = key
        self.prc_params = prc_params
        self.tokenizer = tokenizer
        self.robust_scan = robust_scan

        self._prc = LDPCPRC0(self.prc_params, seed=_seed_to_uint64(self.key))
        self._sk, _pk = self._prc.keygen()

    def _mask(self, block_idx_1based: int) -> int:
        return _mask_from_seed(self.key, block_idx_1based=block_idx_1based, n_bits=self.prc_params.n)

    def _bit_length(self) -> int:
        if not self.tokenizer:
            return 0
        bit_length, _, _ = build_binary_vocab(self.tokenizer)
        return bit_length

    def _bitstring_from_ids(self, token_ids: List[int], bit_length: int) -> str:
        bits = []
        for tid in token_ids:
            for bit_idx in range(bit_length):
                b = (tid >> (bit_length - 1 - bit_idx)) & 1
                bits.append(str(b))
        return "".join(bits)

    def _tokenize_and_binarize(self, text: str) -> str:
        if not self.tokenizer or not text:
            return ""
        bit_length = self._bit_length()
        gen_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self._bitstring_from_ids(gen_ids, bit_length)

    def detect(self, bitstring_or_res: Any) -> Dict[str, Any]:
        if isinstance(bitstring_or_res, dict):
            # Prefer the exact sampled token IDs if available to avoid
            # tokenizer non-invertibility issues (decode->encode mismatches).
            gen_ids = bitstring_or_res.get("generated_ids")
            if isinstance(gen_ids, list) and self.tokenizer:
                bit_length = self._bit_length()
                try:
                    bitstring = self._bitstring_from_ids([int(x) for x in gen_ids], bit_length)
                except Exception:
                    text = bitstring_or_res.get("generated_text", "")
                    bitstring = self._tokenize_and_binarize(text)
            else:
                text = bitstring_or_res.get("generated_text", "")
                bitstring = self._tokenize_and_binarize(text)
        elif isinstance(bitstring_or_res, str):
            bitstring = bitstring_or_res
        else:
            bitstring = ""

        n = int(self.prc_params.n)
        L = len(bitstring)

        if L < n:
            return {
                "detected": False,
                "detection_score": 0.0,
                "num_bits": L,
                "num_windows": 0,
                "hit_start": -1,
                "hit_block": -1,
            }

        # Default (fast) detector: check block-aligned length-n windows.
        # This matches how PRCWatermark embeds blocks sequentially from the start.
        if not self.robust_scan:
            num_full_blocks = L // n
            for block_idx in range(1, num_full_blocks + 1):
                start = (block_idx - 1) * n
                window_bits = bitstring[start : start + n]
                window_int = _pack_bits_lsb_first(window_bits)
                candidate = window_int ^ self._mask(block_idx)
                if self._prc.decode(self._sk, candidate) == 1:
                    return {
                        "detected": True,
                        "detection_score": 1.0,
                        "num_bits": L,
                        "num_windows": num_full_blocks,
                        "hit_start": start,
                        "hit_block": block_idx,
                    }
            return {
                "detected": False,
                "detection_score": 0.0,
                "num_bits": L,
                "num_windows": num_full_blocks,
                "hit_start": -1,
                "hit_block": -1,
            }

        # Robust path: scan all length-n windows and all possible OTP blocks.
        max_blocks = max(1, math.ceil(L / n) + 1)

        windows_checked = 0
        for start in range(0, L - n + 1):
            window_bits = bitstring[start : start + n]
            window_int = _pack_bits_lsb_first(window_bits)
            windows_checked += 1

            for block_idx in range(1, max_blocks + 1):
                candidate = window_int ^ self._mask(block_idx)
                if self._prc.decode(self._sk, candidate) == 1:
                    return {
                        "detected": True,
                        "detection_score": 1.0,
                        "num_bits": L,
                        "num_windows": windows_checked,
                        "hit_start": start,
                        "hit_block": block_idx,
                    }

        return {
            "detected": False,
            "detection_score": 0.0,
            "num_bits": L,
            "num_windows": windows_checked,
            "hit_start": -1,
            "hit_block": -1,
        }

    def detect_batch(
        self,
        results: List[Dict[str, Any]],
        true_labels: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        for idx, res in enumerate(results):
            det = self.detect(res)
            if true_labels is not None:
                det["true_label"] = true_labels[idx]
            res["detection"] = det
        return results

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        tp = fp = tn = fn = 0
        for res in results:
            det = res.get("detection", {})
            pred = int(det.get("detected", False))
            true = det.get("true_label", None)
            if true is None:
                continue
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 0 and pred == 0:
                tn += 1
            elif true == 1 and pred == 0:
                fn += 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = 1.0 - fpr
        fnr = 1.0 - tpr
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": tpr,
            "fpr": fpr,
            "tnr": tnr,
            "fnr": fnr,
            "precision": precision,
            "f1": f1,
            "accuracy": accuracy,
        }
