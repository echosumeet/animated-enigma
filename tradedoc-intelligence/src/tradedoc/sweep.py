"""Confidence-threshold sweep and the straight-through-processing curve.

This is the number an operations owner actually buys: at an agreed error budget, what
share of documents clear without a human touching them. Field-level accuracy on its own
is not decision-relevant -- 97% per field across nine fields is a document that is
right 76% of the time.

Definitions used throughout:
  auto-approve  document confidence (min over fields and the line-item table) >= t
  correct       every field and every line item matches generator ground truth
  precision     P(correct | auto-approved)  -- one minus the escaped-error rate
  recall        P(auto-approved | correct)  -- the work we could have avoided and didn't
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class SweepPoint:
    threshold: float
    stp_rate: float
    precision: float
    recall: float
    error_rate: float
    n_auto: int
    n_auto_wrong: int


def sweep_thresholds(confidences: list[float], correct: list[bool],
                     grid: Optional[np.ndarray] = None) -> list[SweepPoint]:
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    n = len(conf)
    if grid is None:
        grid = np.round(np.arange(0.0, 1.001, 0.02), 4)
    points = []
    total_correct = int(ok.sum())
    for t in grid:
        auto = conf >= t
        n_auto = int(auto.sum())
        n_auto_ok = int((auto & ok).sum())
        precision = n_auto_ok / n_auto if n_auto else 1.0
        recall = n_auto_ok / total_correct if total_correct else 0.0
        points.append(SweepPoint(
            threshold=float(t), stp_rate=n_auto / n if n else 0.0,
            precision=round(precision, 4), recall=round(recall, 4),
            error_rate=round(1.0 - precision, 4), n_auto=n_auto,
            n_auto_wrong=n_auto - n_auto_ok,
        ))
    return points


def operating_point(points: list[SweepPoint], error_budget: float = 0.02
                    ) -> Optional[SweepPoint]:
    """Lowest threshold (hence highest STP) whose escaped-error rate fits the budget.

    Scanning upward from the loosest threshold matters: the error curve is not perfectly
    monotone in a finite sample, and taking the first feasible point rather than the
    global argmax avoids sitting on a lucky bin.
    """
    feasible = [p for p in points if p.error_rate <= error_budget and p.n_auto > 0]
    if not feasible:
        return None
    return max(feasible, key=lambda p: (p.stp_rate, -p.threshold))


def field_accuracy_table(per_doc: list[tuple[str, dict[str, bool]]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-field correctness into a doc-type x field accuracy table."""
    acc: dict[str, dict[str, list[int]]] = {}
    for doc_type, flags in per_doc:
        bucket = acc.setdefault(doc_type, {})
        for name, ok in flags.items():
            cell = bucket.setdefault(name, [0, 0])
            cell[0] += int(ok)
            cell[1] += 1
    out: dict[str, dict[str, Any]] = {}
    for doc_type, bucket in acc.items():
        fields = {k: round(v[0] / v[1], 4) for k, v in sorted(bucket.items())}
        macro = sum(fields.values()) / len(fields)
        out[doc_type] = {"fields": fields, "macro_field_accuracy": round(macro, 4),
                         "n_docs": max(v[1] for v in bucket.values())}
    return out
