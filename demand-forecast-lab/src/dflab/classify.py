"""Demand classification: the ADI / CV-squared quadrants.

Reporting a single average accuracy number across an assortment is the most
common way a forecasting evaluation misleads. A portfolio is a mixture: fast
smooth movers where seasonality is the whole game, and a long intermittent tail
where the only question is whether demand happens at all. Averaging over the
mixture hides the fact that different methods win in different regimes.

Syntetos, Boylan & Croston (2005) give the standard partition:

    ADI  = mean inter-demand interval (periods per demand occurrence)
    CV^2 = squared coefficient of variation of *non-zero* demand sizes

    ADI < 1.32, CV^2 < 0.49  -> smooth
    ADI < 1.32, CV^2 >= 0.49 -> erratic
    ADI >= 1.32, CV^2 < 0.49 -> intermittent
    ADI >= 1.32, CV^2 >= 0.49 -> lumpy

The cut-points come from the cost-based comparison of Croston/SBA against
exponential smoothing in that paper; they are conventions, not physical
constants, and this module keeps them configurable.

Reference
---------
Syntetos, A.A., Boylan, J.E. & Croston, J.D. (2005). "On the categorization of
demand patterns." *Journal of the Operational Research Society* 56(5), 495-503.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ADI_CUT",
    "CV2_CUT",
    "QUADRANTS",
    "DemandProfile",
    "adi",
    "cv_squared",
    "classify_series",
    "classify_panel",
]

ADI_CUT = 1.32
CV2_CUT = 0.49
QUADRANTS = ("smooth", "erratic", "intermittent", "lumpy")


@dataclass(frozen=True)
class DemandProfile:
    """Descriptive statistics used for classification."""

    adi: float
    cv2: float
    quadrant: str
    n_periods: int
    n_nonzero: int
    mean_demand: float
    zero_share: float

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "adi": self.adi,
            "cv2": self.cv2,
            "quadrant": self.quadrant,
            "n_periods": self.n_periods,
            "n_nonzero": self.n_nonzero,
            "mean_demand": self.mean_demand,
            "zero_share": self.zero_share,
        }


def adi(y) -> float:
    """Average demand interval = periods / number of demand occurrences.

    Defined on the observed window: a series with 52 periods and 13 demands has
    ADI 4.0. Returns ``inf`` for an all-zero window.
    """
    arr = np.asarray(y, dtype=float).ravel()
    n_nz = int(np.count_nonzero(arr > 0))
    if n_nz == 0:
        return float("inf")
    return float(arr.size) / float(n_nz)


def cv_squared(y) -> float:
    """Squared coefficient of variation of the non-zero demand sizes.

    Computed on non-zero periods only. Including the zeros would conflate size
    variability with occurrence variability, and ADI already measures the
    latter -- keeping them orthogonal is the entire point of the partition.
    """
    arr = np.asarray(y, dtype=float).ravel()
    nz = arr[arr > 0]
    if nz.size < 2:
        return 0.0
    mu = float(np.mean(nz))
    if mu <= 0:
        return 0.0
    sd = float(np.std(nz, ddof=1))
    return float((sd / mu) ** 2)


def classify_series(
    y, adi_cut: float = ADI_CUT, cv2_cut: float = CV2_CUT
) -> DemandProfile:
    """Classify one series into a Syntetos-Boylan-Croston quadrant."""
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("empty series")
    a = adi(arr)
    c = cv_squared(arr)
    intermittent = a >= adi_cut
    erratic = c >= cv2_cut
    if intermittent and erratic:
        q = "lumpy"
    elif intermittent:
        q = "intermittent"
    elif erratic:
        q = "erratic"
    else:
        q = "smooth"
    n_nz = int(np.count_nonzero(arr > 0))
    return DemandProfile(
        adi=a,
        cv2=c,
        quadrant=q,
        n_periods=int(arr.size),
        n_nonzero=n_nz,
        mean_demand=float(np.mean(arr)),
        zero_share=1.0 - n_nz / float(arr.size),
    )


def classify_panel(
    panel, adi_cut: float = ADI_CUT, cv2_cut: float = CV2_CUT
) -> list[DemandProfile]:
    """Classify every row of a ``(n_series, T)`` panel.

    Classification must be computed on the *training* window only. Using the
    full history leaks test-period behaviour into the reporting grouping, which
    quietly flatters whichever method happens to suit the leaked regime.
    """
    P = np.asarray(panel, dtype=float)
    if P.ndim != 2:
        raise ValueError("panel must be 2-D (n_series, T)")
    return [classify_series(row, adi_cut, cv2_cut) for row in P]
