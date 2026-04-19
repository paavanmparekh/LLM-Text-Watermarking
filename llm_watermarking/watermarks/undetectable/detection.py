import math
from typing import Any, Dict, List, Optional
from .generation import _prf_binary
from .wm_logger import logger
from ...binarizer import build_binary_vocab


class WatermarkDetector:
    def __init__(
        self,
        key: bytes,
        lambda_entropy: float = 5.0,
        tokenizer: Any = None,
    ) -> None:
        self.key = key
        self.lam = lambda_entropy
        self.tokenizer = tokenizer

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
        """Convert text into a pure bitstring via tokenizer round-trip."""
        if not self.tokenizer or not text:
            return ""

        bit_length = self._bit_length()
        gen_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self._bitstring_from_ids(gen_ids, bit_length)

    def detect(self, bitstring_or_res: Any) -> Dict[str, Any]:
        """
        Run stateless detector operating OVER A BITSTRING directly.
        To maintain compatibility, this accepts either a bitstring directly,
        or a result dictionary where it extracts the bitstring internally.
        """
        if isinstance(bitstring_or_res, dict):
            text = bitstring_or_res.get("generated_text", "")
            bitstring = self._tokenize_and_binarize(text)
        elif isinstance(bitstring_or_res, str):
            bitstring = bitstring_or_res
        else:
            bitstring = ""

        L = len(bitstring)

        if L == 0:
            detection = {
                "detected": False,
                "stat": 0.0,
                "threshold": 0.0,
                "anchor": -1,
                "detection_score": 0.0,
                "num_bits": 0,
            }
            logger.info("  NOT detected | empty bitstring")
            return detection

        best_stat = float("-inf")
        best_anchor = -1
        best_n_rem = 0
        best_thresh = 0.0

        # Anchor search: check every bit prefix 'i'
        for i in range(L):
            r_candidate = bitstring[:i+1]
            stat = 0.0
            n_rem = 0

            for j in range(i + 1, L):
                f_val = _prf_binary(self.key, r_candidate, j)
                x_j = int(bitstring[j])
                v_j = f_val if x_j == 1 else (1.0 - f_val)
                if v_j <= 0.0:
                    continue
                
                stat += math.log(1.0 / v_j)
                n_rem += 1

            if n_rem == 0:
                continue

            threshold_i = float(n_rem) + self.lam * math.sqrt(float(n_rem))

            if stat > threshold_i:
                detection = {
                    "detected": True,
                    "stat": stat,
                    "threshold": threshold_i,
                    "anchor": i,
                    "detection_score": stat,
                    "num_bits": n_rem,
                }
                logger.info(f"  WATERMARK DETECTED at bit anchor i={i} | stat={stat:.4f} > thresh={threshold_i:.4f}")
                return detection

            if stat > best_stat:
                best_stat = stat
                best_anchor = i
                best_n_rem = n_rem
                best_thresh = threshold_i

        best_stat_val = best_stat if best_stat > float("-inf") else 0.0
        detection = {
            "detected": False,
            "stat": best_stat_val,
            "threshold": best_thresh,
            "anchor": best_anchor,
            "detection_score": best_stat_val,
            "num_bits": best_n_rem,
        }
        logger.info(f"  NOT detected | best_stat={best_stat_val:.4f} at anchor={best_anchor}")
        return detection

    def detect_batch(
        self,
        results: List[Dict[str, Any]],
        true_labels: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run detect() on every result.
        """
        for idx, res in enumerate(results):
            det = self.detect(res)
            
            if true_labels is not None:
                det["true_label"] = true_labels[idx]
            res["detection"] = det
            
        return results

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        tp = fp = tn = fn = 0
        for res in results:
            det  = res.get("detection", {})
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

        tpr       = tp / (tp + fn)       if (tp + fn) > 0       else 0.0
        fpr       = fp / (fp + tn)       if (fp + tn) > 0       else 0.0
        tnr       = 1.0 - fpr
        fnr       = 1.0 - tpr
        precision = tp / (tp + fp)       if (tp + fp) > 0       else 0.0
        f1        = (2 * precision * tpr / (precision + tpr)
                     if (precision + tpr) > 0 else 0.0)
        accuracy  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

        metrics = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr,
            "precision": precision, "f1": f1, "accuracy": accuracy,
        }
        return metrics
