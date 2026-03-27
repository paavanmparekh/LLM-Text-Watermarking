"""
detection.py — Watermark detector for the Undetectable Watermarking scheme.

Implements Algorithm 4 (Detect_sk) from Christ et al. 2023, Section 4.3:

  For each anchor position i in [L]:
      r^(i) = (x_1,...,x_i)       (fixed to the Phase-1 seed in our impl)
      v_j^(i) = x_j * F_sk(r, j) + (1 - x_j) * (1 - F_sk(r, j))
      stat = Σ_{j=i+1}^{L} ln(1 / v_j^(i))
      if stat > (L - i) + λ * sqrt(L - i): return True

Interpretation of v_j:
  - If watermarked: x_j = 1 iff F_sk(r,j) ≤ p1 → F_sk(r,j) ≈ large when x_j=1
    → v_j = 1 * F_sk ≈ large → ln(1/v_j) ≈ small
  - Under H0 (not watermarked): x_j ~ Bernoulli(p1), F_sk ~ Uniform[0,1]
    → v_j ~ Uniform[0,1] → E[ln(1/v_j)] = 1

  Detection fires when the sum Σ ln(1/v_j) is ABOVE the threshold
  (L-i) + λ√(L-i). This catches the case where a non-watermarked suffix
  looks "too random" relative to a wrong PRF seed — the true watermarked
  suffix gives a SMALL sum. So we also compute the min-stat anchor.

  In practice: we also compute a normalised detection score for plotting/metrics.

Usage
-----
    detector = WatermarkDetector(key=bytes.fromhex(key_hex), lambda_entropy=10.0)
    det = detector.detect(result)          # annotates single result
    results = detector.detect_batch(results, true_labels=[1,1,0,0,...])
    metrics = detector.compute_metrics(results)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from .generation import _prf


class WatermarkDetector:
    """
    Detector for UndetectableWatermark (Algorithm 4, Christ et al. 2023).

    Parameters
    ----------
    key : bytes
        Secret key matching the one used during generation.
    lambda_entropy : float
        Security parameter λ. Must match generation λ.
    threshold_sigma : float
        Detection threshold in units of σ above mean.
        Paper uses λ·√(L-i); we expose this as a separate multiplier.
        Default matches the paper (uses lambda_entropy as the σ multiplier).
    """

    def __init__(
        self,
        key: bytes,
        lambda_entropy: float = 10.0,
        threshold_sigma: float = None,
    ) -> None:
        self.key = key
        self.lam = lambda_entropy
        # threshold_sigma defaults to lambda_entropy (paper's λ√(L-i))
        self.threshold_sigma = threshold_sigma if threshold_sigma is not None else lambda_entropy

    # ------------------------------------------------------------------ #
    #  Core detection (Algorithm 4)                                       #
    # ------------------------------------------------------------------ #

    def detect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Algorithm 4 on a single generation result.

        Parameters
        ----------
        result : dict
            Output from UndetectableWatermark.generate().
            Must contain "bit_trace" key.

        Returns
        -------
        dict with keys:
            detected      : bool — overall detection decision
            anchor_stats  : list[float] — detection stat per anchor i
            best_stat     : float — maximum stat across all anchors
            best_anchor   : int — anchor index giving best_stat
            min_stat      : float — minimum stat (watermarked anchors: small)
            detection_score: float — normalised score for plotting
                            (best_stat - mean_null) / std_null
            num_bits      : int — number of Phase-2 bits tested
        """
        trace = result.get("bit_trace", [])
        L = len(trace)

        if L == 0:
            return _empty_detection()

        anchor_stats: List[float] = []
        detected = False
        best_stat = float("-inf")
        best_anchor = -1

        for i in range(L):
            r = trace[i]["r"]   # PRF seed (same for all entries in our impl)
            stat = 0.0
            for j in range(i + 1, L):
                x_j = trace[j]["x"]
                f_val = _prf(self.key, r, trace[j]["bit_pos"])
                v_j = x_j * f_val + (1 - x_j) * (1 - f_val)
                # v_j == 0 only if (x=1,f=0) or (x=0,f=1) — numerically impossible
                # with HMAC PRF (never exactly 0/1), so no guard needed.
                stat += math.log(1.0 / v_j)

            anchor_stats.append(stat)
            remaining = L - i - 1
            if remaining > 0:
                threshold = remaining + self.threshold_sigma * math.sqrt(remaining)
                if stat > threshold:
                    detected = True
            if stat > best_stat:
                best_stat = stat
                best_anchor = i

        min_stat = min(anchor_stats) if anchor_stats else 0.0

        # Normalised z-score: use anchor i=0 (full suffix) as primary stat
        # E[stat | H0] = L-1,  Var[stat | H0] = L-1 (since Var[ln(1/U)] = 1)
        primary_stat = anchor_stats[0] if anchor_stats else 0.0
        null_mean = L - 1
        null_std = math.sqrt(L - 1) if L > 1 else 1.0
        detection_score = (primary_stat - null_mean) / null_std

        detection = {
            "detected":     detected,
            "anchor_stats": anchor_stats,
            "best_stat":    best_stat,
            "best_anchor":  best_anchor,
            "min_stat":     min_stat,
            "detection_score": detection_score,
            "num_bits":     L,
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

        Requires that each result["detection"]["true_label"] is set
        (call detect_batch with true_labels first).

        Returns
        -------
        dict with: tp, fp, tn, fn, tpr, fpr, tnr, fnr, precision, f1,
                   accuracy, auc_approx (trapezoidal over detection score thresholds).
        """
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
        f1 = (2 * precision * tpr / (precision + tpr)
               if (precision + tpr) > 0 else 0.0)
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

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
        "detected":     False,
        "anchor_stats": [],
        "best_stat":    0.0,
        "best_anchor":  -1,
        "min_stat":     0.0,
        "detection_score": 0.0,
        "num_bits":     0,
    }
