"""Named, independent random streams.

Every stochastic component of the shipment generator draws from its own stream,
derived from a single master seed by hashing the stream name into the spawn key.
Two consequences that matter:

1.  Adding a new source of randomness (say, a port strike) does not shift the
    draws of every existing component, so a regression in a benchmark number is
    a real regression and not a reseeding artefact.
2.  Scenario comparisons are paired by construction: the "baseline" and
    "degraded carrier" worlds see the identical demand, weather and calendar
    draws, and differ only in the stream that was deliberately changed.

This is the common-random-numbers discipline from Law (2015), Ch. 11.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

__all__ = ["StreamBank", "stream_key"]


def stream_key(name: str) -> int:
    """Map a stream name to a stable 63-bit integer.

    ``hash()`` is randomised per interpreter run, so it cannot be used here.
    BLAKE2b gives us a stable mapping that does not depend on Python version.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


class StreamBank:
    """A bank of independent ``numpy`` generators keyed by name.

    Parameters
    ----------
    seed:
        Master seed. Two banks with the same seed produce identical streams.
    """

    def __init__(self, seed: int = 20260101) -> None:
        self.seed = int(seed)
        self._cache: dict[str, np.random.Generator] = {}

    def __call__(self, name: str) -> np.random.Generator:
        return self.get(name)

    def get(self, name: str) -> np.random.Generator:
        """Return the generator for ``name``, creating it on first use."""
        gen = self._cache.get(name)
        if gen is None:
            seq = np.random.SeedSequence(entropy=self.seed, spawn_key=(stream_key(name),))
            gen = np.random.default_rng(seq)
            self._cache[name] = gen
        return gen

    def reset(self, names: Iterable[str] | None = None) -> None:
        """Drop cached generators so the next ``get`` restarts them."""
        if names is None:
            self._cache.clear()
        else:
            for name in names:
                self._cache.pop(name, None)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StreamBank(seed={self.seed}, streams={sorted(self._cache)})"
