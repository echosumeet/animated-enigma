"""Golden-zone / level assignment, priced in the same units as travel.

Slotting tools usually optimise walking distance and then bolt on a rule that
says "put fast movers in the golden zone". That rule loses every argument it
has with the distance objective, because distance has a number and ergonomics
has an opinion.

The fix is to price the level. Every pick has a handling time that depends on
the height of the pick face and the weight of what is being lifted; converting
that time to metre-equivalents at the picker's walking speed puts it on the
same axis as travel, and the slotting search can then trade one against the
other honestly. On the benchmark instance the ergonomic term is roughly a
sixth of the objective - big enough that ignoring it changes the answer, small
enough that it never dominates.

The height model is a piecewise-linear penalty:

* a comfortable band (knee to shoulder) with no penalty,
* a stooping penalty below it, linear in the distance below the band,
* a reaching penalty above it, linear in the distance above the band,
* a fixed step-up/ladder penalty once the face is above the reach limit,
  because that is a discontinuity in method, not a gradual worsening,
* a weight multiplier, because a 2 kg case at 3 m is awkward and a 14 kg case
  at 3 m is an injury.

Rate constants are stated below and are meant to be replaced by whatever a
time study says. The shape - band, linear tails, one step change - is the part
worth keeping; NIOSH's revised lifting equation uses the same structure for its
vertical and distance multipliers (Waters et al. 1993).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ErgonomicModel",
    "golden_zone_order",
]


@dataclass(frozen=True)
class ErgonomicModel:
    """Handling-time penalty by pick-face height and case weight."""

    comfortable_low_m: float = 0.75
    comfortable_high_m: float = 1.55
    stoop_seconds_per_m: float = 4.0
    reach_seconds_per_m: float = 3.5
    step_up_height_m: float = 2.10
    step_up_seconds: float = 6.0
    weight_reference_kg: float = 9.0
    weight_exponent: float = 0.6
    walk_speed_mps: float = 1.25

    def height_of_level(self, level: int, level_height_m: float) -> float:
        """Centre height of the pick face at ``level``."""
        return (level + 0.5) * level_height_m

    def seconds(self, height_m: float, case_weight_kg: float) -> float:
        """Extra handling seconds for one pick at this height and weight."""
        below = max(0.0, self.comfortable_low_m - height_m)
        above = max(0.0, height_m - self.comfortable_high_m)
        base = self.stoop_seconds_per_m * below + self.reach_seconds_per_m * above
        if height_m > self.step_up_height_m:
            base += self.step_up_seconds
        w = max(case_weight_kg, 0.1) / self.weight_reference_kg
        return base * (w**self.weight_exponent)

    def metres(self, height_m: float, case_weight_kg: float) -> float:
        """The same penalty expressed as metre-equivalents of walking."""
        return self.seconds(height_m, case_weight_kg) * self.walk_speed_mps

    def level_penalty_table(
        self, n_levels: int, level_height_m: float, case_weight_kg: float
    ) -> np.ndarray:
        """Metre-equivalent penalty for each level at a given case weight."""
        return np.asarray(
            [
                self.metres(self.height_of_level(l, level_height_m), case_weight_kg)
                for l in range(n_levels)
            ],
            dtype=float,
        )

    def is_golden(self, level: int, level_height_m: float) -> bool:
        h = self.height_of_level(level, level_height_m)
        return self.comfortable_low_m <= h <= self.comfortable_high_m


def golden_zone_order(n_levels: int, level_height_m: float = 1.0) -> list[int]:
    """Levels ordered best-first ergonomically, for a reference case weight.

    Used as the tiebreak when locations are otherwise equidistant from the
    depot. Four levels of a bay share one floor access point, so distance alone
    leaves the choice of level completely undetermined - and a velocity slotting
    that resolves that tie arbitrarily throws away most of the ergonomic gain
    available for free.
    """
    model = ErgonomicModel()
    penalties = model.level_penalty_table(n_levels, level_height_m, model.weight_reference_kg)
    return sorted(range(n_levels), key=lambda l: (penalties[l], l))
