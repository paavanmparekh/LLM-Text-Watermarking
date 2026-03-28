import math
from typing import Any, Dict, List, Optional, Tuple

from .generation import _prf


class WatermarkDetector:
    """
    Detector for UndetectableWatermark — Algorithm 4 (Christ et al. 2023).

    Parameters
    ----------
    key : bytes
        Secret key matching the one used during generation.
    lambda_entropy : float
        Security parameter λ. Must match the value used during generation.
    threshold_sigma : float
        The σ multiplier in the detection threshold:
            threshold = L + threshold_sigma * sqrt(L)
        where L = number of Phase-2 bits being tested.

        The paper uses λ as this multiplier. With λ=5.0 and L≈1000 bits this
        gives threshold = 1000 + 158, which the watermarked score exceeds
        reliably when the model has moderate per-bit entropy. Use smaller
        values (e.g. 2.0–3.0) for short outputs or low-entropy models.

        Default: lambda_entropy (paper-faithful). Override if sequences are
        short and detection fails.
    """

    def __init__(
        self,
        key: bytes,
        lambda_entropy: float = 5.0,
        tokenizer: Any = None,
        threshold_sigma: float = None,
    ) -> None:
        self.key = key
        self.lam = lambda_entropy
        # Default to self.lam so that the detection constraint directly reflects
        # the user's chosen lambda bound as per algorithm theory.
        self.threshold_sigma = threshold_sigma if threshold_sigma is not None else self.lam
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    #  Core detection (Algorithm 4)                                       #
    # ------------------------------------------------------------------ #

    def detect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Algorithm 4 — stateless detector via token ID bits reconstruction.
        """
        gen_ids = result.get("generated_ids")
        all_tokens = result.get("all_tokens")
        
        if not gen_ids or not all_tokens:
            generated_text = result.get("generated_text", "")
            if not generated_text or self.tokenizer is None:
                return _empty_detection()

            gen_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
            all_tokens = self.tokenizer.convert_ids_to_tokens(gen_ids)

        bit_length: Optional[int] = result.get("bit_length")
        N = len(all_tokens)

        if N == 0:
            return _empty_detection()

        # ------------------------------------------------------------------ #
        #  Algorithm 4 — anchor search                                         #
        # ------------------------------------------------------------------ #
        detected    = False
        best_stat   = float("-inf")
        best_anchor = -1
        best_n_rem  = 0

        for i in range(N + 1):
            r_candidate: Tuple = tuple(all_tokens[:i])

            # Score all bits from tokens at step >= i
            stat:  float = 0.0
            n_rem: int   = 0

            for step in range(i+1, N):
                actual_tid = int(gen_ids[step])
                for bit_idx in range(bit_length):
                    x_j     = (actual_tid >> (bit_length - 1 - bit_idx)) & 1
                    bit_pos = step * bit_length + bit_idx
                    f_val   = _prf(self.key, r_candidate, bit_pos)
                    v_j     = f_val if x_j == 1 else (1.0 - f_val)
                    stat   += math.log(1.0 / v_j)
                    n_rem  += 1

            if n_rem == 0:
                continue

            threshold_i = float(n_rem) + self.threshold_sigma * math.sqrt(float(n_rem))
            if stat > threshold_i:
                detection: Dict[str, Any] = {
                    "detected":        True,
                    "best_stat":       stat,
                    "best_anchor":     i,
                    "detection_score": (stat - float(n_rem)) / math.sqrt(float(n_rem)) if n_rem > 0 else 0.0,
                    "num_bits":        n_rem,
                    "threshold":       self.threshold_sigma,
                }
                result["detection"] = detection
                return detection

            if stat > best_stat:
                best_stat   = stat
                best_anchor = i
                best_n_rem  = n_rem


        detection: Dict[str, Any] = {
            "detected":        False,
            "best_stat":       best_stat,
            "best_anchor":     best_anchor,
            "detection_score": (best_stat - float(best_n_rem)) / math.sqrt(float(best_n_rem)) if best_n_rem > 0 else 0.0,
            "num_bits":        best_n_rem,
            "threshold":       self.threshold_sigma,
        }
        result["detection"] = detection
        return detection


    # ------------------------------------------------------------------ #
    #  Batch detection + metrics                                          #
    # ------------------------------------------------------------------ #

    def detect_batch(
        self,
        results: List[Dict[str, Any]],
        true_labels: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run detect() on every result and annotate with 'detection' key.

        Parameters
        ----------
        results : list of dicts
        true_labels : list of int, optional
            1 = watermarked, 0 = not watermarked.
            Enables compute_metrics() afterwards.

        Returns
        -------
        Same results list, each entry annotated with result["detection"].
        """
        for i, res in enumerate(results):
            det = self.detect(res)
            if true_labels is not None:
                det["true_label"] = true_labels[i]
        return results

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute classification metrics over a batch of detected results.

        Requires detect_batch(results, true_labels=[...]) to have been called.

        Returns
        -------
        dict with: tp, fp, tn, fn, tpr, fpr, tnr, fnr, precision, f1, accuracy.
        """
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

        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr,
            "precision": precision, "f1": f1, "accuracy": accuracy,
        }


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _empty_detection() -> Dict[str, Any]:
    return {
        "detected":        False,
        "anchor_stats":    [],
        "best_stat":       0.0,
        "best_anchor":     -1,
        "min_stat":        0.0,
        "detection_score": 0.0,
        "num_bits":        0,
    }
