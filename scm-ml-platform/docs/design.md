# Design notes

## Shape of the system

Five modules, each one a stage of a release path, wired together only by pandas frames:

`datagen` -> `contracts` -> `features` -> `backtest` (+ `decisions`) -> `registry` -> `monitoring`.

No orchestrator, no service, no database. Every artefact is a dataframe or a JSON file
on disk, so any stage can be run, diffed and reasoned about in isolation. That is a
deliberate constraint: the parts of a forecasting platform that actually fail are the
contracts between stages, and those are easiest to inspect when the plumbing is dumb.

## The modelling decisions

**Two independent leakage checks, not one.** The knowledge-time check is declarative:
each source column carries the delay before its value is settled in the source system,
and a spec that reads it sooner is rejected regardless of how causal the dataframe
looks. The truncation check is empirical: rebuild each feature from history truncated
at the label timestamp and assert the value does not move. Neither subsumes the other.
The first misses full-sample group statistics that read only past columns; the second
misses columns that are backfilled late upstream. Both bugs are seeded in
`leaky_specs()` and each check catches exactly one of them.

**The gate has a cost arm.** An accuracy-only gate promotes models that buy wMAPE by
under-forecasting. In the shipped benchmark a challenger whose predictions are scaled
by 0.85 lands at wMAPE 0.2223 against the champion's 0.1865 -- a 3.6 point accuracy
loss, easy to argue about -- but its realised inventory cost per unit is 0.7038 against
0.1995, a 3.5x increase, which is not arguable. Cost is computed by pushing forecasts
through a periodic-review order-up-to policy with a newsvendor critical ratio (Silver,
Pyke & Peterson) and charging holding on the overage and a shortage penalty on the
shortfall.

**Regret against an irreducible baseline.** Reporting absolute cost invites the
question "compared with what". The baseline here is the same safety-stock policy run
with a perfect point forecast, which still carries safety stock and still costs money.
The gap between realised and irreducible cost is the part a better model could actually
recover, and it is the only number worth putting in a business case.

**PSI and KS side by side.** PSI is binned and bounded, so it thresholds cleanly on a
dashboard and an on-call engineer can act on it. KS is distribution-free with a p-value,
which is what a retraining review needs. They disagree often enough -- KS flags
statistically significant shifts that are operationally irrelevant at panel scale -- that
reporting only one hides the disagreement.

**Slices before aggregates.** Aggregate wMAPE of 0.1865 conceals a long-tail category
at 0.3322, 1.78x the overall level. Slice analysis runs on every backtest, not on demand.
