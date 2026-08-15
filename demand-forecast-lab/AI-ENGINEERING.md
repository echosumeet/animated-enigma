# AI Engineering skills map — `demand-forecast-lab`

A benchmarking harness for intermittent and hierarchical demand forecasting. It
generates a seeded 96-series weekly panel, classifies each series into a
Syntetos-Boylan-Croston demand quadrant, runs fourteen methods through a rolling-origin
backtest, reports accuracy per quadrant, and reconciles a 180-node grouped hierarchy.

Mapped against Andrew Ng's four AI Engineering skills, this repository demonstrates
**software engineering fundamentals**, **using coding agents** and **shaping the build**
directly. It does not demonstrate **building and deploying AI applications**: there is
no LLM and no agentic workflow here, and the one learned model in the product is a
gradient-boosted regressor, which is classical ML. What it does carry is the measurement
discipline skill 1 rests on.

## Disclosure

The code in this repository was written by Claude subagents on 2026-08-15, in a single
session, under human direction. The human set the goal, the constraints and the
portfolio composition and made the scope calls; the machine wrote the design, the
implementation, the tests, the benchmark harness, the figures and the README. This
document maps both the product and the build process against the skills map. The suite
is 110 tests, all passing, re-confirmed while writing this file with
`PYTHONPATH=src python -m unittest discover -s tests`.

## 1. Building and deploying AI applications

No LLM, no retrieval, no agent loop runs in this product. `src/dflab/ml.py` fits
`HistGradientBoostingRegressor` — a learned model, but a classical one, and one of
fourteen entries in a zoo otherwise made of state-space smoothing and intermittent
estimators.

The transferable part is the evaluation harness, because the failure this repository is
built around also sinks AI systems: a pooled average that hides where the system is bad.
`src/dflab/backtest.py` scores every method on every series in every window and
aggregates by quadrant. There is no champion — there is a routing rule:

| quadrant | series | share of volume | best method | WAPE | snaive WAPE | FVA |
|---|---|---|---|---|---|---|
| smooth | 28 | 52.4% | `hw_mul[m=52]` | 0.3044 | 0.4586 | 0.336 |
| erratic | 19 | 37.0% | `hw_mul[m=52]` | 0.4296 | 0.5856 | 0.266 |
| intermittent | 20 | 8.5% | `gbt_median` | 0.3070 | 0.3926 | 0.218 |
| lumpy | 29 | 2.0% | `zero` | 1.0000 | 1.4175 | 0.295 |

Forty-nine of the ninety-six series carry 10.5% of the volume, so any pooled metric is a
report on the head. On the lumpy quadrant no real method beats a constant zero forecast
(closest: `gbt_median` at 1.0129), and the harness reports that rather than hiding it —
point accuracy is the wrong objective for that slice.

Three further pieces are what an eval suite needs:

- **A degenerate baseline that self-normalises the scale.** `zero` is in the zoo, and a
  zero forecast scores WAPE exactly 1.0 — anything above 1.0 is worse than giving up.
  Asserted by `test_zero_forecast_gives_wape_of_exactly_one` in `tests/test_metrics.py`.
- **Leakage tests as first-class evals.** `tests/test_ml_and_backtest.py` scrambles
  every value after the forecast origin and asserts the design matrix does not move, and
  that a global model refit on a training window whose future has been replaced with
  garbage produces a bit-identical forecast.
- **Error analysis that changed a conclusion.** `gbt_median` (absolute-error loss) wins
  WAPE at 0.3976 but carries −14.7% bias; `gbt_mean` (squared error) is nearly unbiased
  at −2.3% and loses at 0.4315. The metric that ranks the models is not the one that
  makes them safe downstream.

## 2. Software engineering fundamentals

The environment had no PyPI access — numpy, pandas, scipy, scikit-learn, matplotlib,
networkx and the standard library only. That became a design rule: Croston, SBA, TSB and
the Holt-Winters family are written from the published equations in
`src/dflab/intermittent.py` and `src/dflab/ets.py`, fitted by box-constrained
`scipy.optimize.minimize` with multi-start, not imported from statsmodels.

Tradeoff decisions visible in the code:

- **Two forecaster contracts, dispatched on one flag.** Local models implement
  `fit(y)`/`predict(h)`; the global model implements `fit_panel`/`predict_panel`
  (`src/dflab/base.py`, `src/dflab/ml.py`). Dispatching on `is_global` puts the cut-off
  boundary in exactly one place. The usual way a forecasting benchmark ends up wrong is
  a global model wired into a loop written for local ones.
- **A guard that is a countable diagnostic, not a clamp.** Multiplicative Holt-Winters
  divides by the seasonal index; a zero week makes it explode, and before the guard
  existed `hw_mul` scored WAPE **365** on the intermittent quadrant. `src/dflab/ets.py`
  detects the condition, fits the additive form and sets `degenerate_ = True` — so the
  output is the countable statement that 62 of 96 series (65%) cannot support the
  seasonal model the engine defaults to.
- **Cost against accuracy, measured.** From `benchmarks/results.md`: `ma[w=8]` fits in
  0.3s for WAPE 0.4533, `sba` in 1.1s for 0.4495, `hw_mul[m=52]` in 58.3s for 0.3994.
  The whole benchmark is 323.6s, 306.4s of it the backtest.

Reconciliation carries the sharpest reliability argument: independent base forecasts
miss coherency by 5.99e+03 units at the worst node, and every method drives that to
exactly zero because each `G` satisfies `S G S = S`, asserted in
`tests/test_hierarchy.py`. MinT wins five of eight levels including the bottom (0.4229
vs 0.4399 base), at a fitted Schäfer-Strimmer shrinkage intensity of 0.428.

## 3. Using coding agents

A shared `CONVENTIONS.md` was the context contract handed to every subagent — content
rules, environment constraints, required layout, README specification, definition of
done. That file, not the individual prompts, is why ten independently written
repositories look like one portfolio. Fan-out was one subagent per repository; on two
cores, effective concurrency was about two agents, not ten.

The loop was closed with verifiers rather than assertions: the definition of done
required each agent to actually run its tests, its benchmark and its examples and write
the README from that output. Fabricating a benchmark number was prohibited; figures had
to be generated by running code. A separate audit agent that had written none of the
code re-ran every suite and benchmark across all ten repositories and checked README
numbers against `benchmarks/results.md`.

This repository is also the evidence for the planning-versus-execution pitfall. Against
a 1,200–2,500 line target it landed at 5,448 lines of Python across `src/`, `tests/`,
`benchmarks/` and `examples/` (counted with `wc -l`), and committed second of the ten at
06:42 UTC. Repositories like this one are why the run projected to roughly three hours;
it was stopped after four had completed and the remaining six were relaunched under a
hard budget of 900–1,400 lines. This one was already committed, and was not trimmed.

## 4. Shaping the build

The product decision here is the unit of reporting. A benchmark that returns "method X
wins, WAPE 0.3976" is a worse artifact than one that returns a four-branch routing rule,
even though the second is harder to summarise. Everything downstream follows: quadrants
are computed on the training window only (first 208 periods, before the first forecast
origin) so the grouping cannot leak; MASE denominators are recomputed per window from
training data; both WAPE and MASE are reported because they disagree (`ma[w=8]`: MASE
1.056, WAPE 0.4533 — good on the many small series, poor on the few large), and the
disagreement says which population you are optimising for.

Ownership shows up in what was left in rather than tuned away: the lumpy quadrant losing
to a constant zero, q50 coverage at 0.653, and the whole modelling stack buying only
FVA 0.250 against a seasonal naive that is four lines of numpy.

## AI during development vs AI in the product

| AI during development | AI in the running product |
|---|---|
| Subagents wrote design, code, tests, benchmarks, figures, README | — |
| `CONVENTIONS.md` as context contract across ten agents | — |
| Test, benchmark and example runs as verifiers | — |
| An independent audit agent re-ran everything | — |

The second column is empty. Nothing here calls a model at runtime beyond `sklearn`'s
gradient boosting, and no LLM produces any number this repository reports.

## What I would do differently

1. **Model censoring.** Every zero in the panel is a genuine zero-demand period, but in
   reality many observed zeros are stockouts, and a model fitted to censored history
   under-forecasts exactly the items that need the most stock. A stockout mask and a
   censored-demand estimator is the highest-value extension.
2. **Evaluate the lumpy quadrant on the decision, not on WAPE.** Feed each method's
   forecast into an (R, s, S) policy and compare fill rate at matched holding cost. That
   converts "1.2811 vs 1.2721" into a number somebody can act on.
3. **Budget the repository before writing it.** 5,448 lines against a 1,200–2,500 target
   is a planning failure, and the cost was borne by the six repositories built under a
   hard cap afterwards. A line budget checked at design time would have caught it.

## Takeaways

- **Skill 1.** Per-slice evaluation changes the conclusion, not the presentation. Pooled,
  `gbt_median` is a clean win at 0.3976; split by quadrant it wins one branch of four.
- **Skill 2.** A guard that surfaces a countable flag beats one that silently repairs.
  `degenerate_ = True` turned an unusable WAPE of 365 into the statement "65% of the
  assortment cannot use this model."
- **Skill 3.** Agents close their own loop only if handed a verifier. Three commands
  that had to actually run, plus a ban on fabricated numbers, is what makes the README
  trustworthy; an independent audit agent confirmed it.
- **Skill 4.** The unit of reporting is the highest-leverage decision in a benchmark.
  Everything defensible here descends from deciding the answer is a routing rule, not a
  champion.

## How to explore this repo

Start with `benchmarks/results.md` — every number quoted here is in it. Then
`src/dflab/backtest.py` for the rolling-origin loop and per-quadrant pooling,
`src/dflab/ml.py` for leakage-safe features, and `src/dflab/hierarchy.py` for the
`S G S = S` reconciliation. `tests/test_ml_and_backtest.py` and
`tests/test_hierarchy.py` read as specifications; `docs/design.md` sections 5, 6 and 8
carry the uncomfortable findings.
