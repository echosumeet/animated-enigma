"""Great-circle geometry and continuous facility location primitives.

Distances drive every lane cost in this package, so they are computed once, in
one place, with an explicit road-circuity factor. Straight-line distance
understates road miles by roughly 15-25% depending on terrain and network
density; ignoring that is the most common reason a greenfield study's savings
evaporate when the transport team prices the lanes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "EARTH_RADIUS_KM",
    "haversine_km",
    "haversine_matrix",
    "road_distance_km",
    "center_of_gravity",
    "weiszfeld",
]

EARTH_RADIUS_KM = 6371.0088

#: Typical road-network circuity multiplier applied to great-circle distance.
DEFAULT_CIRCUITY = 1.17


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points (degrees)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def haversine_matrix(
    origins: Sequence[tuple[float, float]], dests: Sequence[tuple[float, float]]
) -> np.ndarray:
    """Vectorised pairwise great-circle distances, shape ``(len(o), len(d))``."""
    o = np.radians(np.asarray(origins, dtype=float))
    d = np.radians(np.asarray(dests, dtype=float))
    lat1 = o[:, 0][:, None]
    lon1 = o[:, 1][:, None]
    lat2 = d[:, 0][None, :]
    lon2 = d[:, 1][None, :]
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def road_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float, circuity: float = DEFAULT_CIRCUITY
) -> float:
    """Great-circle distance inflated by a road-circuity factor."""
    return circuity * haversine_km(lat1, lon1, lat2, lon2)


def center_of_gravity(
    points: Sequence[tuple[float, float]], weights: Iterable[float] | None = None
) -> tuple[float, float]:
    """Weight-weighted arithmetic centroid of lat/lon points.

    This is the textbook "center of gravity" and it minimises weighted squared
    distance, not weighted distance. It is reported here because practitioners
    still ask for it, but :func:`weiszfeld` is the estimator that actually
    minimises transport cost.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        raise ValueError("no points supplied")
    w = np.ones(len(pts)) if weights is None else np.asarray(list(weights), dtype=float)
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive number")
    return float(np.average(pts[:, 0], weights=w)), float(np.average(pts[:, 1], weights=w))


def weiszfeld(
    points: Sequence[tuple[float, float]],
    weights: Iterable[float] | None = None,
    *,
    tol: float = 1e-7,
    max_iter: int = 500,
) -> tuple[float, float]:
    """Weighted geometric median (1-median) by Weiszfeld's algorithm.

    Minimises ``sum_i w_i * d(x, p_i)`` with ``d`` the great-circle distance.
    Weiszfeld (1937); the degenerate case where the iterate lands exactly on a
    demand point is handled by the standard epsilon-guard.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        raise ValueError("no points supplied")
    w = np.ones(len(pts)) if weights is None else np.asarray(list(weights), dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive number")

    cur = np.array(center_of_gravity(pts, w), dtype=float)
    for _ in range(max_iter):
        d = haversine_matrix([tuple(cur)], pts)[0]
        d = np.maximum(d, 1e-6)  # epsilon guard against a point-coincident iterate
        coef = w / d
        nxt = np.array(
            [
                float(np.sum(coef * pts[:, 0]) / np.sum(coef)),
                float(np.sum(coef * pts[:, 1]) / np.sum(coef)),
            ]
        )
        if haversine_km(cur[0], cur[1], nxt[0], nxt[1]) < tol:
            cur = nxt
            break
        cur = nxt
    return float(cur[0]), float(cur[1])
