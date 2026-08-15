# ADR-0003: Central platform, embedded model owners

Status: Accepted

## Context

Two organisational shapes are on the table. A central forecasting team owns every model
and every pipeline; or each planning domain owns its models and the platform team owns
only the shared plumbing.

Fully central concentrates scarce ML skill, produces consistent methodology, and scales
badly: the central team becomes the queue for every domain's roadmap, and it accumulates
context debt because it does not live with the consequences of its forecasts. Fully
embedded gives each domain speed and accountability, and reliably produces five
incompatible feature pipelines, five leakage bugs and no shared definition of accuracy.

The asymmetry that decides it: methodology mistakes are recoverable and visible, whereas
correctness mistakes -- leakage, contract drift, skew -- are silent and expensive. Those
should be owned by whoever can automate them away, which is the platform team.

## Decision

The platform team owns contracts, feature spec semantics, the registry, the gate and
monitoring, and is accountable for correctness. Domain teams own their models, their
feature specs, their thresholds and their forecast quality, and are accountable for the
business outcome. Promotion requires the domain owner's approval and a green gate; the
platform team cannot promote on a domain's behalf and cannot block a green promotion.

## Consequences

- Correctness is enforced once, in CI, for everyone; domain teams cannot opt out of the
  leakage audit or the contract check.
- Domain teams keep roadmap control and cannot blame a central queue for slow delivery.
- The platform team has real power (it defines the gate) with no delivery accountability,
  which is a known failure shape. Mitigation: gate threshold changes require the sign-off
  of an affected domain owner.
- Methodology will diverge across domains. Accepted; a quarterly methods review is the
  forum for convergence, not a mandate.
- Onboarding a new domain costs platform-team time, so the platform roadmap must reserve
  capacity for it rather than treating each onboarding as an interrupt.
