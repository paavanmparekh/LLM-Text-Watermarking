import math
from typing import Any, Dict, List, Optional, Tuple

from .generation import _prf
from .wm_logger import logger


class WatermarkDetector:
    """
    Detector for UndetectableWatermark — Algorithm 4 (Christ et al. 2023).

    Parameters
    ----------
    key : bytes
        Secret key matching the one used during generation.
    lambda_entropy : float
        Security parameter λ. Must match the value used during generation.
        Detection threshold per anchor i:
            threshold = n_rem + λ * sqrt(n_rem)
        where n_rem = number of Phase-2 bits being scored.
        Watermark is declared present when stat > threshold directly.
    """

    def __init__(
        self,
        key: bytes,
        lambda_entropy: float = 5.0,
        tokenizer: Any = None,
    ) -> None:
        self.key = key
        self.lam = lambda_entropy
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    #  Core detection (Algorithm 4)                                       #
    # ------------------------------------------------------------------ #

    def detect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Algorithm 4 — true stateless detector via text re-tokenization.
        """
        if self.tokenizer is None:
            raise ValueError("WatermarkDetector requires tokenizer to operate in true stateless mode.")

        # In the real world, the detector ONLY receives the suspicious text, never the prompt!
        gen_text = result.get("generated_text", "")
        prompt   = result.get("prompt", "")[:60]

        logger.info(f"--- DETECT | prompt='{prompt}...'")

        if not gen_text:
            logger.warning("  Empty generated_text — returning empty detection.")
            return _empty_detection()

        # Tokenize the raw text. Known limitation: if the text was edited or if the tokenizer
        # merges subwords differently without the prompt context, detection may fail (tokenization shift).
        gen_ids    = self.tokenizer.encode(gen_text, add_special_tokens=False)
        all_tokens = self.tokenizer.convert_ids_to_tokens(gen_ids)

        bit_length: Optional[int] = result.get("bit_length")
        N = len(all_tokens)

        logger.info(f"  Tokens={N}, bit_length={bit_length}, lambda={self.lam}")
        logger.debug(f"  Token IDs (first 20): {gen_ids[:20]}")
        logger.debug(f"  Tokens   (first 20): {all_tokens[:20]}")

        if N == 0:
            logger.warning("  N=0 after tokenization — returning empty detection.")
            return _empty_detection()

        if bit_length is None or bit_length == 0:
            logger.error("  bit_length missing or zero in result dict — cannot detect.")
            return _empty_detection()

        # ------------------------------------------------------------------ #
        #  Algorithm 4 — anchor search                                         #
        #  Iterate every candidate anchor i = 0..N.                           #
        #  Detection is binary: if any anchor produces stat > threshold        #
        #  → watermark detected. Return immediately.                           #
        #  If no anchor crosses threshold → not detected, but still report     #
        #  the best stat seen across all anchors for score distribution         #
        #  comparisons between watermarked and unwatermarked text.             #
        # ------------------------------------------------------------------ #

        best_stat   = float("-inf")
        best_anchor = -1
        best_n_rem  = 0
        best_thresh = 0.0

        for i in range(N + 1):
            r_candidate: Tuple = tuple(all_tokens[:i])

            # Score all bits from tokens at position >= i (Phase-2 onward).
            stat:  float = 0.0
            n_rem: int   = 0

            for step in range(i, N):
                actual_tid = int(gen_ids[step])
                for bit_idx in range(bit_length):
                    x_j     = (actual_tid >> (bit_length - 1 - bit_idx)) & 1
                    bit_pos = step * bit_length + bit_idx
                    f_val   = _prf(self.key, r_candidate, bit_pos)
                    v_j     = f_val if x_j == 1 else (1.0 - f_val)
                    if v_j <= 0.0:
                        continue
                    stat  += math.log(1.0 / v_j)
                    n_rem += 1

            if n_rem == 0:
                continue

            threshold_i = float(n_rem) + self.lam * math.sqrt(float(n_rem))

            logger.debug(f"  anchor i={i} | stat={stat:.4f} | threshold={threshold_i:.4f} | bits={n_rem}")

            # Binary detection decision — return immediately on first hit.
            if stat > threshold_i:
                detection: Dict[str, Any] = {
                    "detected":        True,
                    "stat":            stat,
                    "threshold":       threshold_i,
                    "anchor":          i,
                    "detection_score": stat,
                    "num_bits":        n_rem,
                }
                result["detection"] = detection
                logger.info(
                    f"  WATERMARK DETECTED at anchor i={i} | "
                    f"stat={stat:.4f} > threshold={threshold_i:.4f} | bits={n_rem}"
                )
                return detection

            # Track best stat across all anchors for reporting purposes.
            if stat > best_stat:
                best_stat   = stat
                best_anchor = i
                best_n_rem  = n_rem
                best_thresh = threshold_i

        # ---------------------------------------------------------------- #
        #  Not detected — report the best stat seen across all anchors.     #
        #  detection_score is always populated for distribution comparison. #
        # ---------------------------------------------------------------- #
        best_stat_val = best_stat if best_stat > float("-inf") else 0.0
        detection: Dict[str, Any] = {
            "detected":        False,
            "stat":            best_stat_val,
            "threshold":       best_thresh,
            "anchor":          best_anchor,
            "detection_score": best_stat_val,
            "num_bits":        best_n_rem,
        }
        result["detection"] = detection
        logger.info(
            f"  NOT detected | best_stat={best_stat_val:.4f} at anchor={best_anchor} "
            f"| threshold={best_thresh:.4f} | bits={best_n_rem}"
        )
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

        metrics = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr,
            "precision": precision, "f1": f1, "accuracy": accuracy,
        }
        logger.info(
            f"  METRICS | TP={tp} FP={fp} TN={tn} FN={fn} | "
            f"TPR={tpr:.3f} FPR={fpr:.3f} Acc={accuracy:.3f} F1={f1:.3f}"
        )
        return metrics


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _empty_detection() -> Dict[str, Any]:
    return {
        "detected":        False,
        "stat":            0.0,
        "threshold":       0.0,
        "anchor":          -1,
        "detection_score": 0.0,
        "num_bits":        0,
    }
