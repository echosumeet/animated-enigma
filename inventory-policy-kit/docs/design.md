# Design notes

This document covers the architecture and, more usefully, the modelling decisions
and why they went the way they did. The code is deliberately small; the decisions
are where the content is.

## Layering

```
distributions.py      loss functions, quantiles, gamma/normal/empirical/mixture
        |
leadtime.py           lead-time demand: convolution, stochastic lead time, undershoot
        |
safety_stock.py       CSL sizing, fill-rate inversion, empirical quantile sizing
        |
policies.py           (s,Q) (s,S) (R,S) (R,s,S) construction and analytic evaluation
        |
simulation.py         Monte Carlo evaluator - the falsification harness
```

Independent branches hanging off the same distribution layer:

```
newsvendor.py         single-period critical fractile
lotsizing.py          EOQ, discounts, Wagner-Whitin DP, Silver-Meal
serial.py             Clark-Scarf echelon decomposition + serial-system simulator
guaranteed_service.py Graves-Willems tree DP
pooling.py            square-root law and its failure modes
frontier.py           cost-service exchange curves
cli.py                `python -m invkit <command>`
```

The dependency direction is strictly downward. `simulation.py` imports `policies`
only for the `Policy` protocol - it never imports a sizing function, which is what
keeps the analytic-vs-simulated tests from being circular.

## Decision 1: the loss function is the primitive, not the z-table

Every service calculation in this package routes through
`loss(x) = E[(D - x)^+]`. Cycle service is a CDF evaluation; fill rate, expected
backorders, shortage cost and the newsvendor objective are all loss-function
evaluations. Writing it once against a small distribution interface means the
gamma, normal, empirical and mixture cases all get the same formulas for free.

The alternative - a `z` lookup with a normality assumption baked in - is what
makes most planning systems structurally unable to answer "what if the errors
aren't normal?".

## Decision 2: gamma, not normal, as the default lead-time demand model

Three reasons, in order of importance:

1. **Exact convolution.** `Gamma(k, theta)` summed over `L` periods is
   `Gamma(L*k, theta)`. There is no moment-matching error in the lead-time
   aggregation, which means a discrepancy between formula and simulation is a
   real defect rather than an approximation artifact. That is what makes the
   validation tests worth running.
2. **Non-negative.** A normal with CV 0.3 puts 0.04% of its mass below zero. That
   is small but it is not nothing, and it means the simulated process and the
   analytic process are subtly different distributions.
3. **Right-skewed.** Real demand is. A normal understates the upper tail, which
   is the only part of the distribution safety stock is about.

The normal is still available (`NormalLTD`) because the closed-form standard
normal loss is the reference implementation everything else is checked against.

## Decision 3: undershoot is a first-class term, not a footnote

A continuous-review `(s, Q)` policy does not order when the position equals `s`;
it orders when a transaction takes the position *through* `s`. The position at
the moment of ordering is `s - U`, and the reorder point therefore has to cover
`D_L + U`, not `D_L`.

By renewal theory `U` has the equilibrium distribution of the transaction size:

```
E[U]   = E[X^2] / (2 E[X]) = mu/2 + sigma^2 / (2 mu)
E[U^2] = E[X^3] / (3 E[X])
```

Two things follow that are worth internalising:

- On the reference item (mean 100, sd 30 per day) `E[U] = 54.5` units - more than
  half a day of demand, and comparable to the entire safety stock at a 95% cycle
  service target. Omitting it costs **15.7 points** of measured cycle service.
  That number is in `benchmarks/results.md`, measured, not asserted.
- The `sigma^2 / (2 mu)` term is invariant to how finely you slice the period. It
  is set by the coefficient of variation of the *transaction size*, not by how
  often the system polls stock. Moving from a nightly MRP batch to real-time
  inventory visibility does not make undershoot go away; changing how customers
  order does.

This is the single largest source of the gap between what a planning parameter
promises and what the warehouse measures, and it is almost never in the formula.

## Decision 4: timing conventions are stated, because they are the bug

Most published disagreement about inventory formulas is really disagreement about
when things happen inside a period. Two conventions are fixed here and used
consistently:

**Policy simulation** (`simulation.py`): each slice is *receive -> order ->
demand*. This makes the `(s, Q)` exposure window exactly `L` periods and the
`(R, S)` protection interval exactly `R + L`, which is what the formulas assume.
Getting this wrong by one period is a silent 20% error in the reorder point on a
5-day lead time.

**Serial system** (`serial.py`): same sequence, with holding and backorder costs
charged on end-of-period echelon inventory levels, so stage `i`'s risk period is
`L_i + 1`. The Clark-Scarf recursion and the simulator both use it; the fact that
they agree to within 0.5% is evidence that both are right.

**Inventory averaging**: on-hand is averaged over a slice as the midpoint of its
pre- and post-demand values, not sampled at the end. Depletion within a slice is
continuous, so an end-of-slice snapshot understates average on-hand by half a
slice of demand - which is exactly the `mu/2` discrepancy that makes a simulated
holding cost disagree with the textbook `Q/2 + SS`.

## Decision 5: cycle service is measured per cycle, not per period

`SimulationResult` exposes three service measures because they answer three
different questions and conflating them is how a broken reorder point passes
review:

- `cycle_service_level` - fraction of replenishment cycles with no stockout. This
  is the event `P(D_L <= s)` describes. Measured by tracking each open order and
  flagging it if demand goes unmet before it lands.
- `ready_rate` - fraction of periods ending with non-negative net inventory. The
  right comparator for a periodic order-up-to policy with `R = 1`, where every
  period is its own protection interval.
- `fill_rate` - units served from stock over units demanded. The only one the
  business measures.

For `(R, S)` with `R = 5`, the ready rate comes in at 0.989 against a 0.95
per-cycle target - not because anything is wrong, but because only the last
period of the review cycle is fully exposed. Reporting one number and calling it
"service level" hides that entirely.

## Decision 6: Clark-Scarf on a grid, checked against a simulator

The recursion is implemented numerically on an integer grid rather than in closed
form, because the closed form only exists for a normal demand model and the
induced-penalty function is not normal even when demand is.

Cost accounting: echelon holding costs `e_i = h_i - h_{i+1}` charged on echelon
inventory *levels* (which can be negative), plus a penalty of `p + H` on stage-1
backorders where `H = sum_i e_i`. The `H` term is not a fudge - charging `e_i` on
a negative echelon level already credits back holding on backordered units, so it
has to be added back to recover the true cost. Getting this wrong produces a
plausible-looking answer that is quietly 10-20% off, which is why the simulator
exists.

Measured agreement: **-0.45%, -0.45%, -0.33%** for 2-, 3- and 4-stage systems.

Left-edge extrapolation of the cost arrays is linear, which is exact rather than
merely convenient: once you are deep in backorder territory, one more unit short
costs exactly `p + H - e`.

## Decision 7: guaranteed service as a tree DP, checked against brute force

The Graves-Willems model is solved by a dynamic program over integer service
times. The tree is rooted arbitrarily and each node contributes a table
`h[S, SI] = c(SI + T - S) + subtree terms`; supplier children contribute a
prefix-minimum over their outbound service, customer children contribute directly.
Complexity is `O(N * M^2)` in the service-time grid.

`enumerate_optimal_cost` brute-forces the same problem by enumerating every
integer service-time vector. It is exponential and useless in production, which is
exactly what makes it a good oracle: the test suite checks the DP against it on
20 randomly generated trees and requires exact agreement.

What the model actually says on the example BOM: **6 of 9 stages carry the buffer,
3 hold nothing at all**, and the placement is invariant to the service level -
only the scale changes with `z`. That is a structurally different recommendation
from "every stage gets a 95% service level", and it is the reason this model is
what network-design work actually runs.

The bounded-demand assumption is the price of admission. Demand over `t` periods
is assumed never to exceed `mu*t + z*sigma*sqrt(t)`; anything above that is
handled outside the model, by expediting or by a conversation with the customer.
That is a real assumption, it understates the buffer on genuinely fat-tailed
items, and it is stated here rather than buried because the failure mode - a
zero-safety-stock pass-through stage turning out to be the constraint - is a
well-known one.

## Decision 8: no optimisation library

`scipy.optimize.milp` was available and not used. Every optimisation in this
package has structure that makes a purpose-built method both faster and more
transparent:

- Fill-rate inversion: monotone scalar root find. Bisection, ~40 iterations.
- Wagner-Whitin: shortest path on a DAG with the zero-inventory-ordering property.
  `O(T^2)`, exact.
- Clark-Scarf: sequential one-dimensional minimisations by construction.
- Graves-Willems: tree DP, exact for any tree.

An MILP formulation of any of these would be slower, harder to explain, and would
obscure the structural result that makes the method interesting.

## What I would change with more time

- **Vectorise the fill-rate inversion.** 5.5 ms per solve is fine for analysis and
  far too slow for a million-SKU nightly run. The bisection is trivially
  vectorisable across items; the current code solves one at a time.
- **Lost-sales sizing.** Everything analytic here assumes backordering. Lost sales
  changes the recursion, and the current tests only demonstrate that backorder
  parameters over-buffer in a lost-sales world rather than fixing it.
- **Correlated demand across items and periods.** Every model here assumes iid
  demand across periods. Autocorrelation during a demand regime shift is where
  buffers actually fail, and bootstrap aggregation of forecast errors is too
  optimistic in exactly that case.
- **Capacity.** Clark-Scarf and Graves-Willems both assume uncapacitated stages.
  Capacity turns the guaranteed-service problem non-tree-decomposable in general.
