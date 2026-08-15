"""Named, independent random streams -- the machinery behind common random numbers.

The single most common way a simulation study reaches a wrong conclusion is
comparing two configurations that saw different demand. If scenario A and
scenario B each get their randomness from one global generator, then changing
anything structural in B (an extra review epoch, one more lead-time draw)
shifts every subsequent draw and the comparison silently acquires a large
nuisance variance. You then need an order of magnitude more replications to see
an effect you could have seen in ten.

The fix is *variance reduction by common random numbers* (Law, "Simulation
Modeling and Analysis", 5e, Ch. 11): drive each stochastic element from its own
stream, so a structural change in one part of the model cannot perturb the
draws in another. Here a stream is identified by ``(base_seed, replication,
name)`` and nothing else, so:

* replication ``r`` of *every* scenario sees exactly the same customer demand;
* adding a lead-time draw for the factory does not move the retailer's demand;
* re-running a single scenario a year later reproduces it exactly.

The pairing this creates is what makes the paired confidence intervals in
:mod:`echelonsim.metrics` legitimate.
"""

from __future__ import annotations

import zlib
from typing import Dict

import numpy as np

__all__ = ["StreamBank", "stream_key"]


def stream_key(name: str) -> int:
    """Stable 32-bit key for a stream name.

    ``hash()`` is salted per interpreter process, which would make runs
    irreproducible across sessions. CRC32 is not a good hash function but it is
    a perfectly good *stable* one, which is the only property needed here.
    """
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


class StreamBank:
    """A bank of independent ``numpy`` generators keyed by name.

    Parameters
    ----------
    seed:
        The experiment-level seed. Two scenarios compared under CRN must share it.
    replication:
        Index of the replication. Streams for different replications are
        independent; streams for the same replication across scenarios are
        identical.
    """

    def __init__(self, seed: int = 12345, replication: int = 0) -> None:
        self.seed = int(seed)
        self.replication = int(replication)
        self._streams: Dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        generator = self._streams.get(name)
        if generator is None:
            sequence = np.random.SeedSequence(
                entropy=self.seed, spawn_key=(self.replication, stream_key(name))
            )
            generator = np.random.default_rng(sequence)
            self._streams[name] = generator
        return generator

    def child(self, replication: int) -> "StreamBank":
        """A bank for another replication of the same experiment."""
        return StreamBank(self.seed, replication)

    @property
    def stream_names(self) -> tuple:
        return tuple(sorted(self._streams))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StreamBank(seed={self.seed}, replication={self.replication}, streams={len(self._streams)})"
