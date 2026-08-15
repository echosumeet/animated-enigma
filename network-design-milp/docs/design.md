# Design notes

How this library is put together, and why each piece is the shape it is.

## Layering

```
modeling.py            algebraic layer: variables, expressions, sparse assembly, HiGHS
   |
   +-- geometry.py     haversine, centroid, geometric median
   +-- instances.py    data model + synthetic generator + scenario sampler
        |
        +-- facility_location.py   UFLP / CFLP (verifiable by enumeration)
        +-- network_flow.py        the multi-echelon design MILP
             |
             +-- stochastic.py     two-stage deterministic equivalent, VSS / EVPI
             +-- greenfield.py     center-of-gravity heuristic, costed against the MILP
             +-- diagnostics.py    capacity ledger + elastic relaxation
             +-- scenarios.py      scenario runner, sweep, network stability
             +-- reporting.py      cost breakdown, utilisation, service profile
             +-- figures.py        matplotlib (optional import)
```

Nothing below `network_flow` imports anything above it. `figures` is imported
by the figure script only, so `matplotlib` is never on the test path.

## Why a modelling layer at all

`scipy.optimize.milp` takes a dense objective vector, an integrality mask,
bounds, and a `LinearConstraint` over a matrix. Building a four-echelon,
multi-commodity, multi-mode, multi-scenario model directly against that
interface means maintaining an integer offset per variable block by hand.

The failure mode is not that this is tedious. It is that the resulting bugs are
*silent*. An off-by-one in a block offset produces a model that solves happily
and answers a different question. There is no exception, no warning, and the
objective value looks plausible. In a network study that error survives all the
way to a capital request.

So `modeling.py` provides four things and nothing else:

**Variable registry.** `add_vars(keys, name=...)` takes an iterable of
meaningful keys - `("PL02", "DC05", "rail", "SKU-A")` - and returns a
`VarGroup` that maps keys to expressions. Column indices exist but are never
written by hand.

**Wildcard selection.** `VarGroup` builds a positional inverted index over
tuple keys, so `flow.sum(ANY, dc, ANY, sku)` is a set intersection rather than
a scan. Flow balance then reads as:

```python
m.add(flow.sum(ANY, d.id, ANY, c) - flow.sum(d.id, ANY, ANY, c) == 0.0)
```

which is the conservation equation as written on a whiteboard. On the default
instance the arc set is small enough that a scan would do; on the large one
(1,507 lanes x 3 commodities x 4 echelons of selection) the index is the
difference between a build measured in milliseconds and one measured in
seconds.

**Expressions with operators.** `LinExpr` is a sparse dict of coefficients plus
a constant, with `+ - *` and the three comparisons. Comparisons return a
two-sided row with the constant folded into the bound. Multiplying two
expressions raises `TypeError` rather than silently producing nonsense.

**Sparse assembly.** Rows are accumulated as coefficient dicts and assembled
once, at solve time, into COO triplets and then CSR. Objective coefficients can
be attached at variable-creation time, which avoids a second pass over the arc
set purely to build the cost vector.

Deliberately absent: presolve, cut generation, callbacks, column generation.
HiGHS does that work and does it better.

## Big-M, done properly

Three linking constructs appear in the model, and only one of them needs a
big-M:

| construct | form | needs M? |
|:---|:---|:---|
| no flow unless open | `throughput <= M * y` | yes |
| minimum volume if open | `throughput >= min_volume * y` | no |
| arc-level linking | `flow[d,z,c] <= demand[z,c] * y[d]` | the demand *is* the M |

The tight value for the first is `min(capacity, demand that could conceivably
route here)` - `network_flow.throughput_big_m`. `Model.link_big_m` has no
default value for `M`, which forces the modeller to think about it once.

The reason this is worth a paragraph is measurable. On the default instance,
with the same feasible integer set:

| formulation | LP bound | integrality gap |
|:---|---:|---:|
| disaggregated link, tight M | 1,773,829 | 7.50% |
| aggregate link, tight M | 1,722,375 | 10.19% |
| aggregate link, M x100 | 1,019,910 | 46.82% |

The MILP optimum is 1,917,707 in all three. A loose M does not make the model
wrong; it makes the relaxation worthless, so branch and bound has to close the
entire fixed-cost gap by enumeration.

Honest caveat, because the benchmark says so: HiGHS presolve performs
coefficient tightening, and on instances this size it recovers most of the
wall-clock loss. The bound is still the right diagnostic - it is what predicts
behaviour once side constraints (single sourcing, dual sourcing, minimum
volumes) block presolve from seeing through the structure.

## Disaggregated versus aggregated linking

The same effect appears in its textbook form on the DC-to-zone sub-problem.
The strong UFLP formulation states `x_ij <= y_i` per pair; the aggregated one
states `sum_j x_ij <= |J| y_i`. On the generated instance the strong
formulation's LP relaxation is **integral** (gap 0.00%) while the aggregated
one has a 30.02% gap - and both agree with brute-force enumeration of all 255
open sets at 439,554.

That is the classical result (Balinski 1965; Krarup & Pruzan 1983), and
`facility_location.py` keeps both formulations precisely so it can be measured
rather than asserted.

The full network model carries both linking forms: the aggregate throughput row
(which is also the capacity constraint) and the per-arc row. The per-arc form
adds `|DC| x |zone| x |SKU|` rows per scenario - 400 on the default instance -
which is the cost of the tighter bound.

## The first-stage / second-stage split

In the stochastic model, the first stage is:

- open/close for plants and DCs,
- the single-sourcing assignment, when it is enabled.

The second stage is flow, replicated per scenario.

The assignment belongs in the first stage even though it looks like a routing
decision. Single sourcing is a commercial arrangement - one carrier
relationship, one bill of lading, one point of contact - and you do not
renegotiate it when demand comes in 8% high. Putting it in the second stage
produces a model that quietly assumes you can re-source every customer every
period, which flatters the answer.

The deterministic model is the same builder with one scenario at probability 1.
There is no separate code path, which is why the test that a one-scenario
stochastic model reproduces the deterministic model is meaningful.

## Recourse has to exist

`solve_two_stage` forces `allow_unmet=True`. Without an always-available
recourse action, the expected-value design is simply infeasible in the
high-demand scenarios and `EEV = +inf`. That is technically correct and
practically useless: "your plan fails" is not a quantity anyone can trade off
against a fixed cost.

Pricing the shortfall at lost margin turns infeasibility into a number, and the
number is what makes VSS interpretable. The penalty is an input
(`NetworkOptions.unmet_penalty`, default 45 per unit against a ~7.7 per unit
landed cost) and it should be set from gross margin, not guessed.

## Why the scenario generator has three levels

`demand_scenarios` draws `1 + national + regional(metro) + zone`.

Independent per-zone noise diversifies away: across 25 zones the aggregate
coefficient of variation collapses, the stochastic optimum lands on the
mean-value optimum, and VSS comes back as exactly zero. That is the most common
reason a published VSS is zero, and it is an artefact of the sampler rather
than a property of the problem.

What stresses a network is the part that does *not* diversify:

- the **national** factor moves total volume against fixed capacity - it drives
  *whether* to hedge;
- the **regional** factor moves volume between metros - it drives *where*;
- the **zone** factor is included mostly to demonstrate that it does not matter.

`tests/test_stochastic.py` asserts the aggregate spread is more than 3x larger
under correlated shocks than under independent ones at the same total variance.

## Feasibility diagnostics

Two layers, cheap first:

1. **Capacity ledger** - supply, plant capacity, DC capacity and demand, per
   commodity. Most infeasibilities are one number in that table.
2. **Elastic relaxation** - `Model.elastic_copy(tags)` clones the model, adds a
   non-negative slack to every row carrying a listed tag, and minimises total
   violation. Rows that come back non-zero are a near-minimal explanation.

Balance and linking rows are excluded from relaxation on purpose. Relaxing
conservation of flow always "fixes" the model and explains nothing.

This is not a certified IIS. It is the practical payload of one - *what has to
change, and by how much* - and it works with any LP backend. On the stressed
instance in `examples/04` the minimum violation is 61,271.6 units, and solving
the same instance with the shortfall priced in yields 61,271 units unserved. The two agree because they are the same linear programme viewed from
two directions, and that agreement is a useful check on the diagnostic.

## Greenfield as an upper bound, not an answer

The center-of-gravity heuristic is implemented properly - alternating
assignment and weighted geometric median (Weiszfeld), multi-start, weighted by
shipped **kilograms** rather than units - and then priced by solving the flow
problem with its DCs fixed open.

Two things fall out. First, geography alone gets within 1.36% of the MILP
optimum on the default instance, which is the honest reason greenfield studies
persist. Second, the heuristic keeps wanting to add sites because distance
always falls with more of them, while the cost curve turns around at three -
because fixed cost and capacity are exactly what a continuous location model
cannot see.

`scipy.optimize.milp` has no warm-start hook. The heuristic's cost is instead
imposed as an objective cutoff row, which is a valid inequality whenever it
comes from a feasible solution. The benchmark measures whether it helps; on
these instances it does not, and the benchmark says so.

## Testing philosophy

87 of the 88 assertions in the suite check a *property*, not a recorded value:

- physics: conservation, capacity, coverage, closed sites carrying nothing;
- theory: `WS <= RP <= EEV`, LP bound <= MILP optimum, strong formulation
  dominating aggregated, monotone p-median cost in p;
- independent computation: brute-force enumeration of every open set, scalar
  haversine against the vectorised matrix, geometric median against the
  centroid on the objective each is supposed to minimise;
- interface: contradictory options rejected at construction, a constant row
  that cannot hold rejected at build time, probabilities that do not sum to one
  rejected.

A regression test on an objective value tells you something changed. These tell
you what broke.
