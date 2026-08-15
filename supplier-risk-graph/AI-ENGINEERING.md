# AI Engineering skills, as they appear in `supplier-risk-graph`

This repository models a multi-tier supply network as a typed graph and answers three
questions on it: which nodes take the most revenue with them when they fail, which
sub-tier suppliers several apparently-independent tier-1 branches converge on, and what
a fix is worth per dollar spent. The reference instance is 41 parts, 23 suppliers, 35
sites, 74 BOM edges and 89 sourcing edges (`benchmarks/results.md`).

Against Andrew Ng's four AI Engineering skills: this repo demonstrates skills 2, 3 and 4
directly, and skill 1 in a limited form. **No model is trained here.** There is no LLM,
no learned estimator, no fitted parameter — the product is graph analytics
(Hopcroft-Tarjan articulation points via `networkx`) plus Monte Carlo. What carries over
from skill 1 is the part that is not about models: quantifying uncertainty instead of
reporting a point estimate, and running an error analysis on the metric itself instead
of trusting the one that was cheapest to compute.

## Disclosure

The code in this repository was written by Claude subagents on 2026-08-15, in a single
session, under human direction. A human chose the problem, the constraints and the scope;
the machine wrote every line of `src/`, `tests/`, `benchmarks/` and the README. This
document maps both product and build process against the skills map. 35 tests, all pass:

```
PYTHONPATH=src python -m unittest discover -s tests
Ran 35 tests in 0.764s
OK
```

The ten-repository portfolio totals 678 passing tests, re-run by a separate audit agent
that had written none of the code.

## Skill 1 — uncertainty and error analysis, without a model

**Uncertainty quantification.** `src/riskgraph/simulate.py` samples per-site disruption
occurrence, start day and a lognormal recovery duration, then evaluates the availability
kernel on event-boundary intervals so overlapping outages compound instead of averaging
out. Over 2,000 trials on a one-year horizon:

| scenario | P(shortfall) | expected loss ($M) | p95 loss ($M) | mean TTR (d) | p95 TTR (d) |
|---|---|---|---|---|---|
| baseline | 81.4% | 31.4 | 93.3 | 70 | 154 |
| no ramp-up flex | 83.2% | 38.2 | 104.4 | 71 | 154 |
| 4 weeks buffer on MAT-CRIT | 81.4% | 29.8 | 86.4 | 70 | 154 |

The p95 loss is 93.3 against an expected 31.4 — three times the mean — and p95 TTR is
154 days against a mean of 70. Reporting only the expectation under-funds the response
threefold, which is why a model evaluation reports a distribution over slices rather
than one aggregate accuracy number.

**Error analysis on the metric, not the implementation.** The obvious ranking for "which
node matters" is graph centrality. It is cheap and it is wrong here.
`src/riskgraph/spof.py` scores every candidate by re-running `flow.output_fraction` with
that node removed, and keeps `degree_ranking` in the same module so the two orderings
can be compared. Degree centrality agrees with the revenue-at-risk ranking on 3 of the
top 8 nodes — it disagrees on 5 of 8 — and two of the top three revenue-at-risk nodes
are not articulation points at all (`benchmarks/results.md`). The disagreement is
asserted in `tests/test_risk.py::test_ranking_is_by_revenue_not_degree` and
`test_degree_ranking_is_a_different_ordering`.

The same analysis runs on concentration metrics. Tier-1 HHI on the reference instance is
0.163 (about six effective suppliers), supplier HHI 0.062, geographic HHI 0.170 — every
number passes a sourcing review. Meanwhile `SUP-020` sits at depth 3 under 5 of 5 tier-1
branches and carries 100% of the $318.2M portfolio revenue.
`concentration.hidden_dependencies` exists because the standard metric is measurably
blind to the failure it is meant to catch;
`tests/test_risk.py::test_tier1_hhi_looks_healthy_while_the_diamond_is_total` pins it.

One honesty note from the build: the tiebreak in `hidden_dependencies`
(`src/riskgraph/concentration.py:141`) was chosen after observing tie behaviour on
generated instances, not derived from first principles. `docs/design.md` says so, and
gives the rationale — deeper node first, because that is the one nobody is looking at.

## Skill 2 — software engineering fundamentals

The load-bearing decision is **one kernel, three views**. `flow.output_fraction`
(`src/riskgraph/flow.py:106`) answers exactly one question, and SPOF ranking, the Monte
Carlo simulation and the mitigation scorer all call it. Separate availability logic in
the structural and simulated views produces two decks that disagree about the same plant.

Inside that kernel, availability propagates up the BOM as a **minimum, not a product**:
90% of housings and 50% of substrates builds 50% of units, not 45%, and multiplying
would assume components substitute for each other
(`tests/test_flow.py::test_availability_propagates_as_a_minimum_not_a_product`). The
model layer is pydantic-validated and rejects allocation shares that do not sum to one,
dangling site references and BOM cycles (`tests/test_model.py`, first three tests) — bad
master data should surface as a load error, not a plausible-looking risk number.

Tradeoffs are explicit. Removal-scored SPOF ranking costs more than centrality and is
the only actionable ranking. Mitigation scoring re-runs the baseline at each action's
trial count and seed (`src/riskgraph/mitigation.py:114`) under common random numbers,
because a baseline drawn at a different budget mixes Monte Carlo error into the ranking —
the 800-trial baseline is $31.8M against $31.4M at 2,000 trials. The whole benchmark
suite runs in 3.0s on one core.

Dependency discipline was forced by the environment: the build container had no PyPI
access, so the only imports are numpy, scipy, networkx, pydantic, matplotlib and the
standard library, and tests are stdlib `unittest` (still collectable by pytest).

## Skill 3 — using coding agents

One subagent owned this repository end to end: design, implementation, tests, benchmarks,
figure, README. Three mechanisms did the supervision.

*Context contract.* A shared `CONVENTIONS.md` — not the individual prompt — specified
content rules, layout, README structure and the definition of done. Ten independently
written repos look like one portfolio because of it.

*Verifiers, so the agent closes its own loop.* Done required actually running
`python -m unittest discover -s tests`, `python benchmarks/run_benchmarks.py` and
`python examples/tier3_review.py`, and writing the README from real output. Fabricating
a benchmark number was prohibited; every README table is reproducible at seed 7.

*Generalisation check.* A single-instance result is an anecdote, so the harness runs 5
seeds (7, 11, 23, 41, 57) and reports the planted diamond recovered as the top-ranked
hidden dependency in 5/5 (`benchmarks/results.md`), regression-tested in
`tests/test_risk.py::test_planted_diamond_in_generated_network_is_top_ranked`.

## Skill 4 — shaping the build

- **Cut on purpose.** Probabilistic cascade propagation and any live event-ingestion
  adapter — stated in the README, not left as an implied gap.
- **Protected.** The hidden-dependency detector leads because it answers the question a
  sourcing organisation cannot, not because it is the most interesting algorithm.
- **Output framed as a decision.** `mitigation.py` ranks by risk reduction per dollar:
  pre-qualifying MAT-CRIT returns $12.79 per $1 of annual cost, buffer stock $4.20, dual
  sourcing $1.06. Buffer stock removes more absolute risk ($1,381,557 vs $1,151,412);
  pre-qualification removes more per dollar. That ordering is the argument a category
  team on a fixed budget actually has to settle.
- **Budget under time pressure.** This repo was in the build's second batch, after the
  first four repositories overran (near 6,000 lines each against a 1,200–2,500 line
  target). The relaunch imposed 900–1,400 lines, 20–35 tests, one figure, one example.
  This repo lands at 1,589 lines across `src/` and `tests/`, 35 tests,
  `docs/network_spof.png` and `examples/tier3_review.py`. The correction cut scope, not
  verification.

## AI during development vs AI in the product

| AI during development | AI in the running product |
|---|---|
| Claude subagent wrote all source, tests, benchmarks, figure, README | No LLM, no trained model, no learned parameters |
| Agent ran the suite and benchmarks itself; README written from real output | Monte Carlo (`simulate.py`) and graph analytics (`spof.py`, `concentration.py`) |
| Audit agent re-ran everything, checked README against `benchmarks/results.md` | Randomness is sampling, not inference: seeded `np.random.default_rng` |

The second column is statistical and graph computation, not machine learning. No model
is fitted anywhere in this repository.

## What I would do differently

1. **Correlate disruptions.** Occurrence is sampled independently per site, but real risk
   is regional — a typhoon takes a province, a tariff takes a country. Independent
   sampling understates the tail, so the p95 of 93.3 is optimistic in the direction that
   matters most.
2. **Bound ramp-up by capacity.** `flex` is a scalar defaulting to 1.25, but
   `weekly_capacity` is already on the site model and should cap it per site. Baseline
   versus no-flex ($31.4M vs $38.2M expected loss) shows how much this parameter moves.
3. **Solve mitigation as a portfolio.** Actions are scored one at a time; selecting under
   a budget is a knapsack over interacting actions, and greedily taking the top of the
   list is not the same answer.
4. **Instrument coverage.** The metric worth watching is the fraction of tier-2/3 spend
   with a confirmed site-level source, not the risk score. Nothing here computes it.

## Takeaways

- **Skill 1.** The cheap metric and the correct metric disagreed on 5 of the top 8 nodes.
  That number exists only because both were computed and compared; shipping centrality
  because it is standard would have given a confidently wrong ranking.
- **Skill 1.** A p95 loss three times the mean is not a footnote — it is the number the
  decision runs on.
- **Skill 2.** One shared kernel behind three views makes a class of contradiction
  structurally impossible rather than unlikely.
- **Skill 3.** An agent given executable verifiers and a 5-seed generalisation check
  produces claims you can re-derive: `benchmarks/run_benchmarks.py` regenerates every
  README table.
- **Skill 4.** Under a budget cut, cutting features and keeping the eval harness was the
  right trade — 35 tests and a 5-seed benchmark survived, the feature list shrank.

## How to explore this repo

1. `src/riskgraph/flow.py` — `output_fraction`, the kernel everything else calls.
2. `src/riskgraph/concentration.py` — `hidden_dependencies`, the headline capability.
3. `tests/test_risk.py` — the assertions pinning the claims, particularly
   `test_ranking_is_by_revenue_not_degree` and
   `test_tier1_hhi_looks_healthy_while_the_diamond_is_total`.
4. `benchmarks/results.md` — every number quoted above.
5. `examples/tier3_review.py` — the workflow in one script.
