"""Transit-time distributions.

Lead time is parameterised by **mean and coefficient of variation**, not by
shape and scale. That is not cosmetic. The whole point of the lead-time
experiment in this package is to separate "the lead time is long" from "the
lead time is unreliable", and you can only do that cleanly if one knob moves
the mean with CV fixed and the other moves CV with the mean fixed.

Distributions provided:

* :class:`Deterministic` -- the control.
* :class:`GammaLeadTime` -- a shifted gamma with a hard minimum. Gamma is the
  usual choice for a positive, right-skewed duration (Silver, Pyke & Thomas,
  "Inventory and Production Management in Supply Chains", 3e, Ch. 7); the shift
  encodes the physical floor below which no shipment can arrive.
* :class:`DiscreteLeadTime` -- an explicit pmf. Real supplier lead times are
  frequently bimodal ("it comes in 3 days, or it misses the vessel and comes in
  24"), and moment-matching a bimodal duration to a unimodal shape gets the
  tail wrong in exactly the region that safety stock is sized on.

Order crossing (a later order arriving before an earlier one) is *permitted*
under stochastic lead times, and :class:`NoCrossingWrapper` exists to switch it
off. Whether crossing can physically occur depends on the mode -- it can for
parcel, it cannot for a single vessel -- and the choice materially changes
pipeline variance, so it is exposed rather than assumed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "LeadTime",
    "Deterministic",
    "GammaLeadTime",
    "DiscreteLeadTime",
    "NoCrossingWrapper",
]


class LeadTime(ABC):
    """A sampler for transit duration, in periods."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float:
        ...

    @property
    @abstractmethod
    def mean(self) -> float:
        ...

    @property
    @abstractmethod
    def std(self) -> float:
        ...

    @property
    def cv(self) -> float:
        return self.std / self.mean if self.mean > 0 else 0.0

    def reset(self) -> None:
        """Clear state between replications (only stateful wrappers need this)."""


@dataclass
class Deterministic(LeadTime):
    duration: float = 2.0

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.duration)

    @property
    def mean(self) -> float:
        return float(self.duration)

    @property
    def std(self) -> float:
        return 0.0


@dataclass
class GammaLeadTime(LeadTime):
    """Shifted gamma with mean ``mean_lt`` and coefficient of variation ``cv_lt``.

    The shift ``minimum`` is subtracted before fitting, so the realised mean is
    ``mean_lt`` and the realised standard deviation is ``mean_lt * cv_lt``,
    provided ``mean_lt > minimum``. With ``cv_lt = 0`` this degenerates to
    deterministic, which keeps the CV sweep continuous at its left endpoint.
    """

    mean_lt: float = 4.0
    cv_lt: float = 0.3
    minimum: float = 0.5

    def __post_init__(self) -> None:
        if self.mean_lt <= self.minimum and self.cv_lt > 0:
            raise ValueError("mean lead time must exceed the physical minimum")
        if self.cv_lt < 0:
            raise ValueError("cv must be non-negative")

    def sample(self, rng: np.random.Generator) -> float:
        if self.cv_lt <= 0:
            return float(self.mean_lt)
        excess_mean = self.mean_lt - self.minimum
        sigma = self.mean_lt * self.cv_lt
        shape = (excess_mean / sigma) ** 2
        scale = sigma ** 2 / excess_mean
        return float(self.minimum + rng.gamma(shape, scale))

    @property
    def mean(self) -> float:
        return float(self.mean_lt)

    @property
    def std(self) -> float:
        return float(self.mean_lt * self.cv_lt)


@dataclass
class DiscreteLeadTime(LeadTime):
    """Explicit pmf over durations -- the honest model for a bimodal supplier."""

    values: Sequence[float] = (2.0, 8.0)
    probabilities: Sequence[float] = (0.85, 0.15)

    def __post_init__(self) -> None:
        self._values = np.asarray(self.values, dtype=float)
        self._probs = np.asarray(self.probabilities, dtype=float)
        if self._values.shape != self._probs.shape:
            raise ValueError("values and probabilities must have the same length")
        total = self._probs.sum()
        if not np.isclose(total, 1.0):
            raise ValueError(f"probabilities must sum to 1, got {total:.6f}")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.choice(self._values, p=self._probs))

    @property
    def mean(self) -> float:
        return float(np.dot(self._values, self._probs))

    @property
    def std(self) -> float:
        second = float(np.dot(self._values ** 2, self._probs))
        return float(np.sqrt(max(0.0, second - self.mean ** 2)))


@dataclass
class NoCrossingWrapper(LeadTime):
    """Forbid order crossing by clamping each arrival to be no earlier than the last.

    Stateful: it remembers the most recent scheduled arrival time relative to
    the shipment sequence, so it must be reset per replication. Under this
    wrapper the *realised* mean lead time is longer than the base mean -- an
    honest consequence, not a bug: a shipment that would have overtaken waits.
    """

    base: LeadTime = field(default_factory=lambda: GammaLeadTime())
    _elapsed: float = field(default=0.0, repr=False)
    _last_arrival: float = field(default=-np.inf, repr=False)

    def reset(self) -> None:
        self._elapsed = 0.0
        self._last_arrival = -np.inf

    def observe_dispatch(self, time: float) -> None:
        self._elapsed = time

    def sample(self, rng: np.random.Generator) -> float:
        raw = self.base.sample(rng)
        arrival = self._elapsed + raw
        if arrival < self._last_arrival:
            arrival = self._last_arrival
        self._last_arrival = arrival
        return max(0.0, arrival - self._elapsed)

    @property
    def mean(self) -> float:
        return self.base.mean

    @property
    def std(self) -> float:
        return self.base.std
