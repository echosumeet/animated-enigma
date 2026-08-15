# ADR-0004: wMAPE plus signed bias as the accuracy pair

Status: Accepted

## Context

The candidates are MAPE, wMAPE, RMSE, MASE and a pinball loss on quantile forecasts.

MAPE is undefined at zero demand and explodes on low-volume items, which is exactly the
long tail where we most need a number. RMSE is scale-dependent and cannot be aggregated
across a portfolio with three orders of magnitude of volume. MASE (Hyndman & Koehler,
2006) is scale-free and defensible, but it is a ratio to a naive benchmark and planners
consistently misread it; the communication cost is high. Pinball loss is the right
metric if the model produces quantiles, which this one does not yet.

wMAPE weights error by volume, which matches how inventory cost accrues, is defined at
zero demand, and is the metric planning organisations already read. Its weakness is that
it is symmetric and unbounded above, so a systematically low forecast can look
competitive -- which is precisely the failure the cost arm of the gate exists to catch.

No single accuracy metric detects direction. Bias must be reported alongside.

## Decision

wMAPE is the headline accuracy metric. Signed bias, normalised by total demand, is
reported and gated alongside it and is never omitted from a summary. Both are gate arms.
Realised inventory cost (ADR context: `docs/metrics.md`) is the tiebreaker for promotion.

MASE is computed for method comparison in research contexts and is not a gate arm.
Pinball loss is added when quantile forecasts ship.

## Consequences

- The metric matches how planners already talk, so results need no translation.
- Long-tail items are under-weighted by construction. Slice-level reporting exists
  specifically to stop that becoming invisible; the benchmark long-tail slice runs at
  1.78x the overall wMAPE.
- A model can improve wMAPE and worsen bias. That is a legible failure with an explicit
  gate arm rather than a debate.
- Cross-portfolio comparison against published literature is harder without MASE, which
  is why it is still computed even though it does not gate.
