# AI Engineering, as this repository actually demonstrates it

`eta-risk-engine` predicts freight arrival times, attaches an interval whose coverage is
measured rather than asserted, calibrates a delay probability, converts it into one of
three operational actions through an explicit cost matrix, and checks whether its features
have drifted. There is a learned model in the running product, so the AI angle here does
not have to be manufactured.

Mapped against Andrew Ng's AI Engineering Skills Map (14 Aug 2026), it carries evidence
for skill 1 (the classical-ML and evaluation half), skill 2 and skill 3. Skill 4 appears
only in limited form: the decisions here are about what belongs in the pipeline, not
about a real customer. There are no LLMs or agentic workflows in the product, so the
context-engineering and RAG material in skill 1 is absent.

## Disclosure

The code in this repository was written by Claude subagents on 2026-08-15, in a single
session, under human direction. The human (Sumeet) chose the problem, the constraints and
the portfolio composition and made the correction decisions below; the subagent wrote the
design, implementation, tests, benchmark harness, figure and README. This document maps
both product and build process against the four skills. 35 tests, all passing — re-run
while writing this file: `Ran 35 tests in 1.990s / OK`.

Nothing here has run in production, at scale, or at any company. Every shipment is
synthetic (`src/etarisk/generate.py`).

## The four skills, as they actually appeared

### 1. Building and deploying AI applications

All numbers below come from `benchmarks/results.md`, on 60,000 generated shipments split
temporally 36,000/12,000/12,000:

| Claim | Number |
| --- | --- |
| Point ETA MAE vs quoted transit | 25.71 h vs 27.12 h |
| Coverage at nominal 0.900 / 0.950 | 0.889 / 0.943, mean width 92.9 h / 130.0 h |
| ECE, raw → isotonic | 0.0863 → 0.0492 (Brier 0.2245 → 0.2203) |
| Cost per shipment, fixed rule → isotonic model | 248.87 → 215.30 (13.5% saved) |
| Oracle cost per shipment | 79.86 (model captures 19.9% of the gap) |
| PSI, `dest_congestion_obs`, train vs test | 3.37 (verdict: broken) |

**Uncertainty is measured, not claimed.** `ConformalETA` in `src/etarisk/model.py`
implements split conformal per Lei et al. (2018) with the `ceil((n+1)(1-alpha))/n`
quantile; `coverage()` reports empirical against nominal. Coverage lands 1–2 points
*under* nominal at every level (0.777/0.800, 0.889/0.900, 0.943/0.950), and the undershoot
is left in with its cause: the generator injects a regime shift at day 430, and split
conformal is exchangeability-based, so the calibration block predates it.

**Calibration is separated from ranking.** `DelayRiskModel` in `src/etarisk/risk.py` fits
the classifier on the training block and the isotonic map on a temporally *later* block.
Halving ECE while Brier moves 0.0042 is the point: the ordering barely changes, but
probabilities move across the decision threshold.

**A probability is converted into money.** `src/etarisk/decision.py` holds a 3-action ×
2-outcome `CostMatrix` (late: 600/430/150, on-time: 0/80/360) and publishes the implied
indifference thresholds via `CostMatrix.thresholds()` — p = 0.320 to notify, p = 0.500 to
expedite. That is what makes the 2.6-point cost difference between uncalibrated and
calibrated probabilities visible; ECE alone would not have told you it mattered.

**The evaluation protocol is defended by a test.** In `tests/test_features.py`,
`test_every_validation_index_is_after_every_training_index` fails the moment anyone
reintroduces a random split, and `test_in_sample_encoding_leaks_but_out_of_fold_does_not`
asserts that in-sample target encoding correlates above 0.4 with pure-noise labels while
the out-of-fold version stays below 0.10 — the difference between a leaked MAE and an
honest one.

The headline — 5% MAE over the published carrier quote — is unflattering, and was left in
and explained (the quote is built from the same physics; customs holds and the disruption
term are unobservable at booking) rather than tuned against a weaker baseline.

### 2. Software engineering fundamentals

The build container had no PyPI access, so the importable surface was numpy, pandas,
scipy, scikit-learn, matplotlib and the standard library. That drove real choices:
`HistGradientBoosting*` instead of lightgbm, stdlib `unittest` instead of pytest (still
pytest-collectable), a hand-written PSI in `src/etarisk/drift.py` with a categorical
fallback for near-discrete columns.

The structural decision worth naming is `src/etarisk/pipeline.py`: `run_pipeline` is the
only place the six modules are wired together, so `benchmarks/run_benchmarks.py`,
`examples/control_tower.py` and `src/etarisk/cli.py` exercise one code path and cannot
disagree about a number. `CONTRIBUTING.md` makes that a rule and declares the
evaluation-protocol tests load bearing. The pipeline runs in 4.2 s on one core, so the
benchmark is something a reviewer will actually run.

### 3. Using coding agents

The agent was given a shared `CONVENTIONS.md` as its context contract, and a definition of
done requiring it to actually run, here:

```
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python benchmarks/run_benchmarks.py
PYTHONPATH=src python examples/control_tower.py
```

and to write the README from the real output. Fabricating a benchmark number was
prohibited; figures had to come from running `src/etarisk/figures.py`. That is the
supply-the-verifier pattern, and it is why `results.md` and the README agree.

It did not fully work. A separate audit agent, which had written none of the code, re-ran
everything and found this README claiming an 11-second pipeline while its own benchmark
output said 4.2 seconds. Corrected to the measured value. A self-verifying agent still
needs an independent verifier: the agent that wrote the prose is least likely to notice it
drifted from the artifact.

### 4. Shaping the build

Partial. The genuine product decisions here are ordering decisions: protocol before point
model, intervals before probability, cost matrix before any threshold, drift last — the
order in which these things fail on an exception desk, not the order in which they are
interesting. `docs/design.md` names what was left out (no scoring service, no quantile
regression, no multivariate conformal) because none of it changes a conclusion. What is
missing is business context: the cost-matrix numbers are plausible freight figures chosen
by the agent, not elicited from anyone.

## Build narrative for this repository

| Step | What happened | Skill |
| --- | --- | --- |
| Spec | `CONVENTIONS.md` handed to one subagent owning the repo | 3 |
| Design | Four stages fixed in `docs/design.md`; partial observability built into the generator so the conformal work would mean something | 1, 4 |
| Implementation | 1,523 lines in `src/etarisk`, largest module `generate.py` at 615 | 2 |
| Verification | 35 tests, benchmark and example run in-session; README from real output | 1, 3 |
| Correction | Audit agent caught the README's 11 s claim; fixed to the measured 4.2 s | 3 |
| Commit | One commit, `Sumeet <benstokesipl@gmail.com>`, honest date | 4 |

The portfolio-wide scope cut — the first run overshot a 1,200–2,500 line target at ~6,000
lines per repo, and the remaining repos were relaunched at 900–1,400 lines, 20–35 tests,
one figure, one example — is visible in this repository's shape. The record does not say
which cohort this repo fell into, so I am not claiming one. Separately, an initial request
to back-date commits across several years was declined and withdrawn; the dates here are
honest.

## AI during development vs AI in the product

| AI during development | AI in the product |
| --- | --- |
| Subagent wrote design, code, tests, benchmarks, figure, README | Boosted-tree point ETA (`model.py`) |
| Audit agent re-ran suites, caught the 11 s claim | Split-conformal intervals (`model.py`) |
| Verifier commands in the definition of done | Isotonic-calibrated risk model (`risk.py`) |
| Orchestrator, ~2 concurrent agents on 2 cores | Cost matrix (`decision.py`), PSI drift (`drift.py`) |

No LLM runs in this product at any point.

## What I would do differently

1. Use Mondrian conformal, split by mode, so coverage is conditional rather than marginal.
   The horizon-bucket MAE table (7.96 h at <2d against 53.05 h at 14–30d) shows how uneven
   the error is; one global quantile cannot serve both.
2. Implement adaptive conformal (Gibbs & Candès, 2021). The 0.889/0.900 undershoot is a
   known consequence of the regime shift, and the fix is a known method.
3. Make the decision layer a constrained allocation. Expedite capacity is finite, so the
   day's book is a knapsack, not 12,000 independent threshold tests. The model expedites
   47.2% of shipments against the fixed rule's 32.8% — not a purchasable plan.
4. Replace PSI with a two-sample test. Its 0.10/0.25 convention has no significance theory
   behind it; `drift.py` admits that in its own docstring, which is not a fix.
5. Diff README numbers against `benchmarks/results.md` mechanically in the audit pass.

## Takeaways

- **Skill 1.** Reporting coverage below nominal (0.889 vs 0.900) is worth more than a
  method that claims 0.900 and never checks.
- **Skill 1.** ECE moving 0.0863 → 0.0492 is abstract; the same change moving cost per
  shipment 221.82 → 215.30 is not. Attach a cost matrix or you cannot tell which metric
  gains matter.
- **Skill 2.** One wiring point (`pipeline.py`) plus a rule that README numbers come only
  from the benchmark script is what keeps an agent-written repo auditable.
- **Skill 3.** Verifier commands got the code right and the prose wrong; the 11 s / 4.2 s
  defect was caught only by an agent that had not written the code.
- **Skill 4.** The decision that mattered was ordering (protocol → point → interval →
  probability → decision → drift), not model choice. Swapping the regressor moves MAE by
  tenths of an hour; skipping the cost matrix costs the deployment.

## How to explore this repo

1. `benchmarks/results.md` — every number here, in source form.
2. `tests/test_features.py` — the leakage guard.
3. `src/etarisk/decision.py` — the cost matrix and its thresholds.
4. `src/etarisk/model.py` — `ConformalETA` and the `sqrt(planned transit)` scaling.
5. `examples/control_tower.py` — one shift on the exception desk, end to end.
