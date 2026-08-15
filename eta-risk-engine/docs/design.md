# Design notes

## Shape of the problem

Four stages, ordered by what fails first in an operation rather than by what is
interesting to model: evaluation protocol, point estimate, uncertainty,
decision. Each is a module of a few dozen lines composing through plain
`pandas` frames. `etarisk.pipeline.run_pipeline` is the only place they are
wired together, so the benchmarks, the example and the CLI cannot disagree
about a number.

## Data generation

`generate.py` builds transit time as a multiplicative physical core (distance /
mode speed, times lane, carrier and lognormal noise factors) plus a set of
strictly non-negative additive terms: weekend dwell, hub queueing at origin and
destination, weather above a severity threshold, lognormal customs holds on
cross-border lanes, and a truncated Lomax disruption term with tail index 2.2.
Non-negativity is the structural reason the distribution is right-skewed: a
shipment can be arbitrarily late but not arrive before its physical minimum.

Observability is deliberately partial. Congestion is visible as of the booking
day but the index that matters is the one on the arrival day; weather is a
forecast whose noise grows with the square root of the horizon; customs and
disruption are not visible at all. A generator whose target is fully learnable
would make the conformal section meaningless. The default configuration also
injects a regime shift at day 430 — a congestion step change plus two carriers
degrading — so that the calibration and drift sections have something real to
detect.

Each stochastic component draws from its own named stream (`rng.py`), so adding
a source of randomness does not reshuffle the others.

## Modelling decisions and why

**Absolute-error loss.** With a Pareto-ish tail, squared error spends model
capacity on a handful of disrupted shipments. MAE loss targets the conditional
median, which is what an exception desk wants, at the price of a systematic
negative bias against the mean. That tradeoff is stated in the README rather
than hidden.

**Out-of-fold target encoding.** Lane and carrier mean delay is the strongest
single feature and the easiest place to leak. Encodings are computed on
expanding temporal folds, and downstream blocks use an encoder fitted on the
whole training block — the nightly-refit analogue.

**Scaled split conformal.** One quantile in hours across a network spanning
20-hour road moves and 900-hour ocean moves produces useless intervals.
Residuals are normalised by the square root of planned transit before the
quantile is taken. The guarantee remains marginal; conditional coverage by
horizon is measured and reported, not claimed.

**Isotonic on a later block.** The classifier is fitted on the training block,
the isotonic map on the calibration block that follows it in time. That absorbs
part of the drift, and recalibrating needs no retraining.

**Explicit cost matrix.** Three actions, two outcomes, six numbers, and the
implied action thresholds printed alongside. The decision layer is where a model
becomes auditable to someone who does not care how it was fitted.

## What is deliberately absent

No scoring service, no quantile regression, no multivariate conformal. Each adds
surface area without changing a conclusion above.
