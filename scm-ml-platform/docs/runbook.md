# Runbook: degraded forecasting model

Audience: on-call engineer, no prior context on the model.
Goal: reach a decision -- roll back, hold, or escalate -- within 30 minutes.

## 0. Severity

| Level | Condition | Response |
| --- | --- | --- |
| Sev-1 | Forecast pipeline produced no output, or output failed contract validation, and an order cycle runs within 12 hours | Page immediately, roll back on failure to fix within 1 hour |
| Sev-2 | Fill rate below target two consecutive days, or feature PSI > 0.25 on a feature the model depends on | Page during business hours, decide within one working day |
| Sev-3 | Slice wMAPE above 1.25x overall for two consecutive weeks, or override rate up sharply in one category | Ticket, retraining backlog |

If an order cycle is inside the next 12 hours, prefer rolling back over investigating.
The previous version was good enough last week and the investigation will still be
there afterwards.

## 1. Establish what is serving

```bash
PYTHONPATH=src python - <<'PY'
from scmplatform.registry import ModelRegistry
reg = ModelRegistry("var/registry")
card = reg.production("demand_daily")
print(card.version, card.created_at, card.training_window, card.metrics)
print("previous:", reg.previous_production("demand_daily"))
PY
```

Record the version and the previous version before changing anything. If the production
version changed in the last 48 hours, a recent promotion is the prime suspect and step 5
is the fast path.

## 2. Rule out the inputs first

Most "the model is broken" incidents are the feed, not the model.

```bash
PYTHONPATH=src python -m scmplatform validate
```

- **Contract failures present.** This is a data incident, not a model incident. Hand off
  to the feed owner named in the contract. Do not retrain on a bad feed.
- **Freshness violation only.** The feed is late. Confirm the upstream job; the forecast
  is stale but not wrong.
- **Clean.** Continue.

Then check for a silent interface change:

```bash
PYTHONPATH=src python -m scmplatform audit
```

Any point-in-time finding on the production feature set means a spec changed and should
never have shipped. Roll back and open a post-incident review on the CI gate that
allowed it.

## 3. Check drift

```bash
PYTHONPATH=src python -m scmplatform monitor
```

Read in this order:

1. **Prediction drift.** If predictions moved and no input moved, the model is being
   served features it was not trained on. This is a skew bug. Roll back.
2. **Feature drift.** PSI above 0.25 on a feature the model leans on is a real regime
   change. The model is not broken; it is out of date. Retrain, do not roll back --
   rolling back gives you an older model facing the same new world.
3. **PSI between 0.10 and 0.25.** Note it, do not act on it alone.

## 4. Check the slices

Aggregate error can be flat while one category is badly wrong. `slice_performance` runs
in the same command. A single flagged slice with a plausible cause -- a new item cohort,
a promotion, a regional event -- is usually a Sev-3, not a rollback. Flagged slices
across most categories at once point at a global cause, which means go back to step 2.

## 5. Decide

**Roll back when:** a promotion in the last 48 hours preceded the degradation; a skew or
leakage finding is present; the cause is unknown and an order cycle is imminent.

```bash
PYTHONPATH=src python - <<'PY'
from scmplatform.registry import ModelRegistry
reg = ModelRegistry("var/registry")
print(reg.rollback("demand_daily", note="INCIDENT-<id>: <one-line cause>").version)
PY
```

Verify the production version changed, confirm the next scheduled run picks it up, and
post the version and note in the incident channel. Target: under 15 minutes from
decision.

**Hold and retrain when:** inputs are clean, drift is genuine and material, and no
recent promotion correlates. Rolling back into a changed world makes things worse.

**Escalate to the model owner when:** the cost metric moved but accuracy did not, or the
other way round. That pattern usually means the cost parameters or the order policy
changed, which is outside the on-call engineer's remit.

## 6. Close out

- Record in the incident: production version before and after, the check that first
  fired, time to detect, time to decide.
- If the platform did not detect this before a human did, that gap is the finding, not
  the model. Add the missing check.
- If a rollback happened, the superseded version stays archived until the model owner
  has an explanation. Do not re-promote it to clear an alert.
