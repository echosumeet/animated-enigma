# Design notes

## The metric is the whole problem

Every number a slotting tool produces is a function of its distance metric, so
that is where the modelling effort goes. Euclidean distance between two pallet
positions is wrong by a factor that is not constant: two faces across a rack are
2.5 m apart in the plane and sixty metres apart for a picker, who has to walk to
a cross aisle and back. A tool built on the plane will happily slot two
co-ordered SKUs "next to each other" on opposite sides of a rack wall.

`slotting.layout` therefore uses the shortest path on the aisle graph:

    same aisle       -> |y_p - y_q|
    different aisle  -> |x_p - x_q| + min_c (|y_p - c| + |c - y_q|)

with `c` over the cross-aisle y-coordinates. The second term collapses to
`|y_p - y_q|` whenever a cross aisle lies between the points, which is exactly
what a mid-warehouse cross aisle is bought for. `tests/test_layout.py` checks the
closed form against Dijkstra on an explicit `networkx` corridor graph for every
pair of locations in a small warehouse, in one-block and two-block layouts. The
closed form is kept because the search calls it millions of times.

Vertical travel is deliberately excluded. A picker does not walk up the rack;
reaching level 3 costs seconds and shoulder load, not metres. That lives in
`slotting.ergonomics` and is converted to metre-equivalents at a stated walking
speed, so one objective can carry both without pretending height is distance.

## Fit on one period, score on the next

Slotting policies see the first 60% of the order stream; travel is measured on
the last 40%. This is not ceremony. A study that fits and scores on the same
demand reports the fit, not the benefit. On the headline instance the honest
number is 7.7% worse, and that gap is what a real re-slot decays by between
refresh cycles.

The baseline is a random *feasible* fill, reported at the median of several
seeds. Not "slotted by SKU id", which correlates with nothing and so looks worse
than random; not a worst case, which nobody actually has.

## Surrogate objective, exact routing reference

Local search cannot call a router in its inner loop, so `slotting.objective`
scores an assignment with a linear velocity-distance term plus a quadratic
affinity term, and `slotting.calibration` fits those weights *differentially* —
against measured travel around a velocity-slotted reference, at the margin
rather than on levels, since only the marginal rate matters to a swap decision.
The fit reports R² 0.99 on move deltas, and prices affinity at zero here: with
3.5-line orders there is not enough co-occurrence mass for pairing to beat
proximity-to-depot.

Routing is the opposite: the exact answer is cheap, so heuristics are held to
it. `exact_aisle_dp` is the polynomial Ratliff & Rosenthal dynamic program, and
`held_karp` runs independently on every tour small enough for it and is asserted
to agree to 1e-6 — two exact methods, so the gap table does not rest on one
implementation being right.

## What that buys

The result falls out of the two together: slotting changes which router you
should use. S-shape sits 17% above optimal in an unslotted warehouse and 49%
above it once picks concentrate near the depot, because traversing a full aisle
to reach three bays at its mouth is pure waste. Optimising slotting and routing
separately, which is how most sites are organised, leaves that on the floor.
