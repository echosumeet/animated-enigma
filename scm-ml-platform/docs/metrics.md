# Metric tree: from model quality to business outcome

The tree exists to answer one question in a review: *if this number moves, what does it
cost, and who acts on it?* Every level has an owner and a decision attached. A metric
with neither is a chart, not a metric.

```
                      Working capital and service
                                   |
                +------------------+------------------+
                |                                     |
      Forecast-attributable                     Planner trust
        inventory cost                        (override rate)
                |                                     |
     +----------+----------+                          |
     |                     |                          |
 Realised cost      Irreducible cost           Override value-add
 (order policy)     (perfect forecast,
     |              same service target)
     |
 +---+--------------------+
 |                        |
Over-order units    Under-order units
(holding)           (shortage, fill rate)
     |                        |
     +-----------+------------+
                 |
        Forecast error, by slice
                 |
     +-----------+-----------+
     |                       |
   wMAPE                Signed bias
 (magnitude)           (direction)
     |                       |
     +-----------+-----------+
                 |
      Input health and stability
                 |
   +-------------+-------------+-------------+
   |             |             |             |
Contract     Feature drift  Prediction    Point-in-time
violations   (PSI, KS)      drift         findings
```

## Level 1 -- business outcome

**Working capital and service level.** Owned by planning. Not gated by this platform;
it is the reason the platform exists. Moves on a quarterly rhythm and has many causes
other than the forecast, which is why it is never used to evaluate a model release.

## Level 2 -- decision quality

**Forecast-attributable inventory cost** (`decisions.DecisionQuality.regret`). Realised
cost of the order-up-to decisions the forecast produced, minus the cost a perfect point
forecast would still incur under the same service target. This is the only number in the
tree that belongs in a business case, because it is the only one that isolates the part
a better model could recover.

- Owner: platform team reports it; planning owns the cost parameters.
- Decision: a sustained rise triggers a retraining review, not a page.
- Benchmark reading: realised 15,548 against irreducible 3,529, so 4.41x; regret is
  0.1542 cost units per unit of demand.

**Planner trust**, proxied by override rate (ADR-0005). Slow-moving, and the metric that
actually determines whether the platform is used.

## Level 3 -- decision components

**Over-order units** drive holding cost; **under-order units** drive shortage cost and
set the fill rate. Splitting them is what makes a cost regression diagnosable: a
symmetric accuracy loss and a one-sided bias loss produce very different splits, and
only the second is usually urgent.

- Decision: fill rate below the service target for two consecutive weeks is an incident.

## Level 4 -- forecast error, by slice

**wMAPE** for magnitude, **signed bias** for direction (ADR-0004). Always reported per
slice as well as in aggregate, because the aggregate is the number that lies. A slice
above 1.25x the overall wMAPE with enough rows to be meaningful is flagged
automatically.

- Owner: the domain model owner.
- Decision: both are gate arms; a flagged slice enters the retraining backlog.
- Benchmark reading: overall wMAPE 0.1865, long-tail slice 0.3322, ratio 1.78.

## Level 5 -- input health

These are leading indicators. They move before the error does, which is the entire point.

**Contract violations.** Binary and blocking. Any error-severity violation stops the run.

**Feature drift (PSI and KS).** PSI below 0.10 is stable, 0.10 to 0.25 warrants
investigation, above 0.25 pages on-call. KS is reported for the retraining review, not
for alerting -- at panel scale it flags shifts that are statistically real and
operationally irrelevant.

**Prediction drift.** Checked first when inputs look clean, because a prediction
distribution that moved without an input that moved usually means the model is being
served the wrong features, not that the world changed.

**Point-in-time findings.** Must be zero. Not a threshold, a gate.

## How the levels connect

The tree is read downward during an incident and upward during a review. Downward:
cost went up, which component, which slice, which input. Upward: this feature drifted,
does it reach a slice that matters, does that slice move cost enough to act. A level
that cannot be traversed in either direction is not part of the tree.
