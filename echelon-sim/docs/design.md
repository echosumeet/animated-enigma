# Design notes

This document covers the architecture and, more usefully, the modelling decisions
and why they went the way they did. The code is small; the decisions are where
the content is.

## Layering

```
engine.py        heapq event loop: Event, Timeout, Process, Environment, Interrupt
      |
rng.py           named independent streams -- common random numbers
      |
demand.py        customer demand processes        leadtime.py   transit distributions
forecast.py      moving average, exp. smoothing   policies.py   base-stock, (s,S), (R,S), batching
      |
network.py       nodes, echelons, inventory position, information modes
      |
simulation.py    the period structure, wired onto the engine
      |
metrics.py       MSER-5 truncation, batch means, paired and ratio intervals
      |
experiments.py   configs, replication under CRN, scenario comparison
      |
bullwhip.py      information.py      disruption.py      tradeoffs.py      cli.py
```

The dependency direction is strictly downward and one edge is deliberately
missing: `simulation.py` does not import `bullwhip.py`. The simulator has no
knowledge of the closed-form results it is validated against, which is what makes
the validation tests meaningful rather than circular.

---

## Decision 1: hand-roll the event engine

A general-purpose DES package would have worked. Two things argue against it.

**Tie-breaking is a modelling decision, not an implementation detail.** Within
one simulated period a dozen things happen at the same timestamp: trucks arrive,
customers demand, nodes allocate, orders are placed, statistics are recorded.
Which happens "first" changes the answers. A receipt that lands before the
allocation step is available to ship this period; one that lands after is not,
and measured fill rate moves by roughly a lead time's worth of service with no
other change. In `engine.py` that ordering is an explicit `priority` field, and
the whole period structure is a seven-line table in the docstring of
`simulation.py`. In a framework it is spread across the framework's internals.

**Reproducibility has to be exact.** The heap is keyed on
`(time, priority, insertion_counter)` with a strictly increasing counter, so the
ordering is a total order and two runs with the same seeds replay identically.
That is a precondition for common random numbers, and CRN is what makes the
scenario comparisons in this package possible at 30 replications rather than 300.

The cost is about 330 lines. The programming model is the familiar one — a
process is a generator that yields events — plus `Interrupt`, because a supply
disruption is naturally modelled as something that happens *to* a node in the
middle of its normal cycle rather than as a flag the normal cycle has to check.

## Decision 2: the protection interval is `R + L - 1`, and the reason is the phase table

The textbook expression for a periodic-review order-up-to policy is `R + L`.
This package uses `R + L - 1`, and it is not a typo.

The textbook derivation assumes the review happens *before* the period's demand
is served. Here allocation runs at priority 30 and review at priority 50, so by
the time a node reviews it has already shipped today's orders. Today's demand is
therefore not at risk, and the risk period is one period shorter.

Concretely, for `R = 1`: an order placed at the end of period `t` arrives at the
start of `t + L`. Net inventory at the end of any period is `S` minus the last
`L` demands, so the risk period is `L`, which is `R + L - 1`. For general `R` the
low point sits at the end of period `t + R + L - 1` and the same algebra applies.

Getting this wrong costs a full period of mean demand in every target — 100 units
on a 100-unit-per-period item, 400 units of system inventory across four echelons,
bought to fix an arithmetic convention. `Node.protection_interval` carries the
derivation; `TestTimingConvention.test_oracle_base_stock_reproduces_demand_exactly`
pins it, by asserting that a policy with a known mean and no batching places
orders exactly equal to the demand it just served. Any off-by-one in the phase
ordering breaks that identity immediately.

## Decision 3: `on_order` and `in_transit` are separate fields

The obvious implementation carries a single `outstanding` = everything ordered
and not yet received. Installation inventory position does not care about the
split. Echelon inventory position does, and getting it wrong deadlocks the chain.

Echelon inventory position for a node is *physical* stock in its subtree plus its
own outstanding order. Units a downstream node has ordered but not been shipped
are not a second copy of anything — they are still sitting on this node's shelf,
or they do not exist at all because this node is short. Count them and the
echelon position is inflated by the internal order book; every node then sees a
position already at target and nobody ever orders again. Units already dispatched
between two members of the subtree are different: physically real, on nobody's
shelf, counted exactly once.

This was a real bug during development, and its symptom was not a crash — it was
a VMI configuration that produced zero orders, 0% fill rate and a plausible-looking
inventory number. `TestEchelonAccounting` pins each half of the rule separately.

## Decision 4: echelon targets cover the *cumulative* protection interval

The second half of the same bug. An echelon target is compared against echelon
stock, which already contains the downstream inventory — so it has to be large
enough to fund that downstream inventory as well as cover this node's own lead
time. Setting every echelon target to its local `R + L - 1` makes the upstream
targets equal to the downstream ones, the intermediate nodes carry no
installation stock at all, and the chain starves.

`SupplyNetwork.cumulative_protection` sums the protection interval from a node
down the longest path to a retailer. The resulting factory amplification under
VMI matches the single-stage closed form evaluated at that cumulative interval to
within a few percent, which is a satisfying independent confirmation that the
state variable and the target are consistent with each other.

## Decision 5: measure variance, not coefficient of variation

Bullwhip is reported as `Var(orders)/Var(demand)`. The more common dashboard
metric is a ratio of coefficients of variation, and it is misleading here because
it confounds amplification with a change in mean throughput — which is exactly
what happens during a disruption recovery, when the mean order rate rises while
the backlog is worked off. A CV ratio makes a recovering chain look calmer than
it is.

Two versions are reported and they answer different questions. **Local**
amplification (against the node's own incoming order stream) answers "am I making
it worse". **Cumulative** amplification (against true end-customer demand) answers
"how distorted is the signal by the time it reaches me". Every node in the
measured chain has a modest local factor of about four or five. The factory still
sees eighty times the variance of the demand that caused it, because modest
factors multiply. Nobody is behaving badly.

## Decision 6: decompose with Shapley values, not sequential ablation

Demand signal processing, order batching and lead time do not compose additively
and do not commute. Turn one off and subtract, and the answer depends on the
order you turned them off in — which is a property of your procedure, not of the
supply chain.

So all `2^3` combinations are simulated under common random numbers and each
mechanism is credited with its Shapley value on the *log* of the amplification
ratio: log because the mechanisms compose roughly multiplicatively, Shapley
because it is the unique attribution that is efficient (the parts sum to the
whole), symmetric, and independent of ablation order. Efficiency is asserted in
code rather than trusted, because a mis-indexed subset would silently
redistribute attribution without changing the total.

The result is worth the machinery. **Lead time on its own contributes nothing** —
with a perfect forecast and no batching, orders equal demand no matter how long
the lead time is. A one-at-a-time ablation scores lead time at zero and sends the
reader off to fix something else. Its Shapley value is 0.44 in log terms, a
factor of 1.6, *entirely* from interaction with forecasting: a long lead time
does not amplify anything by itself, it multiplies whatever the forecast is
already doing.

The intervals come from computing the Shapley value **per replication** and
taking a t interval across replications. That is only legitimate because all
eight cells are paired by CRN.

## Decision 7: three information modes, differing in exactly two things

| mode | forecasts from | manages |
|---|---|---|
| decentralised | the orders it receives | its own installation stock |
| shared POS | end-customer demand | its own installation stock |
| VMI | end-customer demand | echelon stock (Clark & Scarf) |

Separating "what you know" from "what you control" is the whole point. Sharing
POS data removes the *forecast* cascade; it does not remove the *physical*
cascade, because each echelon still holds and replenishes its own stock on its
own review cycle.

The measured outcome is not a clean ranking, and that is the interesting part.
Shared POS gives the largest reduction in order variance but leaves upstream
buffers thin: each node now sizes its safety stock from the dispersion of *end
demand* while actually facing the dispersion of the *order stream* above it,
which is four times larger. VMI leaves more order variance at the factory —
the echelon target covers the cumulative lead time and so responds harder to a
forecast revision — but places stock where the demand is and comes out cheapest
on total cost.

The comparison is run at a **calibrated service level**: `z` is bisected per mode
until every mode delivers the same retailer fill rate, and only then is inventory
compared. Comparing inventory across modes at different service levels is the
easiest way to overstate an information-sharing business case, and it is what
most of them do.

## Decision 8: warm-up truncation is estimated, and estimated on averaged paths

MSER-5 (White 1997; Franklin & White 2008) picks the truncation point that
minimises the estimated standard error of the truncated mean. It is applied to
the **replication-averaged** system inventory path, not to each pilot separately.

On a single run the statistic is noisy enough to return zero on a series with an
obvious transient — it is trading a genuine variance reduction against the
`1/(n-d)` penalty for discarding data, and on one path the noise wins. Averaging
first collapses the noise by `sqrt(pilots)` without touching the transient, which
is common to every replication because they all start from the same empty
pipeline. Same reasoning as Welch's procedure; the difference is that MSER-5 then
picks the point without anyone having to look at a chart.

The same truncation is applied to *every* scenario in a comparison. A
per-scenario truncation would make the scenarios cover different stretches of
simulated time, which is a subtle way to compare two things that are not
comparable.

## Decision 9: batch means, and reporting the diagnostic with the interval

Consecutive periods of an inventory series are strongly dependent, so the naive
standard error `s/sqrt(n)` understates the true one — by about 50% on the
configuration in `benchmarks/results.md`, and by much more on a smoother series.
`batch_means_ci` blocks the run into 10-30 large batches whose means are
approximately independent (Schmeiser 1982) and **reports the lag-1 correlation of
those batch means in the interval's `note` field**. An interval whose assumption
is checked and reported is worth something; one whose assumption is asserted is
not.

## Decision 10: `allow_returns` exists so the closed form can be tested

Real chains cannot return goods, so the default clamps orders at zero. The
analytic bullwhip results assume signed orders. Keeping it as a flag lets the
test suite check the simulator against the closed form (agreement within 0.5% on
six configurations) while every experiment runs the realistic case.

Enabling it turned out to be worth more than the test. A negative order is a
**purchase-order cancellation**, and the interesting constraint is that you
cannot recall a truck: cancellation bites against the supplier's *unshipped*
backlog, newest line first, and whatever has already been dispatched stays
dispatched and stays on the customer's pipeline. That constraint is why the
simulated amplification falls below the closed form at `alpha = 0.8` — the model
is more realistic than the formula there, not less accurate.

---

## What the model does not do

- **Backorders, never lost sales.** Every unfilled order waits. Lost sales change
  the fill-rate definition, the recovery metric, and the optimal policy shape.
- **Single sourcing.** Each node has one supplier. Dual sourcing is the most
  common real resilience lever and is not modelled.
- **No node exploits the allocation rule.** `Allocation.PROPORTIONAL` creates the
  incentive to inflate orders — Lee, Padmanabhan & Whang's rationing game — but
  every policy here orders its true requirement. Three of their four causes are
  measurable in this package; the rationing game is not.
- **Capacity is enforced but not anticipated.** A capacitated node ships no more
  than its capacity, and no policy knows its own capacity. This is realistic and
  it is also a limitation: a capacity-aware policy would order earlier.
- **Stationary demand parameters.** The forecasters adapt, but the underlying
  process does not drift except in `SeasonalTrend`.
- **One product.** No shared capacity, no allocation across SKUs, no substitution
  — and shared constrained capacity across a portfolio is where a lot of real
  variance amplification actually comes from.
