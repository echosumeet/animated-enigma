# Supply chain engineering portfolio

Ten open-source projects spanning demand planning, inventory theory, network
optimization, simulation, applied ML, and LLM systems — built as one coherent body of
work rather than ten unrelated experiments.

Everything runs on synthetic data the projects generate themselves. Clone this repo,
pick any directory, and you can reproduce every number in its README in under a minute.
**678 tests across the ten projects, all passing.**

Each project also lives as a standalone repository, linked below, for anyone who wants
just one of them.

## Projects

### AI engineering

| Project | Tests | What it is |
|---|---|---|
| [`supply-chain-copilot`](supply-chain-copilot) · [repo](https://github.com/echosumeet/supply-chain-copilot) | 35 | Tool-calling agent over a planning star schema. Guarded text-to-SQL, typed tool registry, execution tracing, and a 25-question golden-set eval harness. Runs end to end offline with no API key. |
| [`tradedoc-intelligence`](tradedoc-intelligence) · [repo](https://github.com/echosumeet/tradedoc-intelligence) | 40 | Trade document extraction where the ground truth is exact, because the repo generates the invoices and bills of lading it then extracts. Hybrid rules plus LLM, with a straight-through-processing curve at a fixed error budget. |
| [`eta-risk-engine`](eta-risk-engine) · [repo](https://github.com/echosumeet/eta-risk-engine) | 35 | Shipment ETA and delay risk with conformal prediction intervals, isotonic calibration, drift detection, and a cost-matrix decision layer that reports value in dollars rather than AUC. |
| [`scm-ml-platform`](scm-ml-platform) · [repo](https://github.com/echosumeet/scm-ml-platform) | 35 | Data contracts, point-in-time-correct features with a skew detector, model registry, backtest gating in CI. Ships with a PRD, six ADRs, a metric tree, and an on-call runbook. |

### Planning and optimization

| Project | Tests | What it is |
|---|---|---|
| [`demand-forecast-lab`](demand-forecast-lab) · [repo](https://github.com/echosumeet/demand-forecast-lab) | 110 | Forecasting benchmark that answers which method wins where — Croston/SBA/TSB, exponential smoothing, gradient boosting, scored per demand quadrant, with MinT hierarchical reconciliation. |
| [`inventory-policy-kit`](inventory-policy-kit) · [repo](https://github.com/echosumeet/inventory-policy-kit) | 121 | Cycle service level versus fill rate — they are different numbers and the gap is the point. Stochastic lead times, Wagner-Whitin, Graves-Willems guaranteed service. Analytic results validated against Monte Carlo in the tests. |
| [`network-design-milp`](network-design-milp) · [repo](https://github.com/echosumeet/network-design-milp) | 88 | Facility location and multi-commodity flow as real MILPs on HiGHS, plus a two-stage stochastic variant reporting VSS and EVPI. |
| [`warehouse-slotting`](warehouse-slotting) · [repo](https://github.com/echosumeet/warehouse-slotting) | 36 | Slotting by velocity and affinity, with pick-path heuristics benchmarked against exact DP routes so each heuristic's real gap is visible. |

### Simulation and risk

| Project | Tests | What it is |
|---|---|---|
| [`echelon-sim`](echelon-sim) · [repo](https://github.com/echosumeet/echelon-sim) | 143 | A discrete-event engine written from scratch, driving bullwhip decomposition, information-sharing regimes, and disruption recovery — with common random numbers and warm-up truncation. |
| [`supplier-risk-graph`](supplier-risk-graph) · [repo](https://github.com/echosumeet/supplier-risk-graph) | 35 | N-tier supplier mapping that finds the diamond dependencies tier-1 diversity hides, ranks single points of failure by revenue at risk, and scores mitigations per dollar. |

## Running any of them

Python 3.10+ with numpy, pandas, scipy, scikit-learn, matplotlib, networkx, pydantic,
and reportlab. Nothing else is required — no pytest, no simpy, no pulp, no ortools,
and no LLM SDK.

```bash
cd supply-chain-copilot
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python benchmarks/run_benchmarks.py
```

## How these were built

Each project carries an `AI-ENGINEERING.md` mapping its work to Andrew Ng's AI
Engineering Skills Map, including an open account of how the code was produced: written
by Claude subagents in a single directed session, against a shared conventions spec,
with automated verifiers and an independent audit pass. That disclosure is deliberate.

Five of the ten contain no AI in the running product at all — they are classical
operations research and statistics, and their documents say so rather than inventing an
angle.

Read `AI-ENGINEERING.md` in any project directory for the specifics.

## A note on results

Several projects report a result where the sophisticated method loses. Local search in
`warehouse-slotting` lands 1.0% worse on travel distance than plain ABC slotting. The
affinity weight there calibrates to exactly zero. The ETA model in `eta-risk-engine`
beats the carrier's own quote by only 5%.

Those were left in and explained rather than tuned away. Negative results that survive
contact with a benchmark are more informative than a table where every proposed method
wins.

## License

MIT.
