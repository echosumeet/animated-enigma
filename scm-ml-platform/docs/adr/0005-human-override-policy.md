# ADR-0005: Overrides are always allowed and always recorded

Status: Accepted

## Context

Planners override system forecasts. Any policy that pretends otherwise is a policy that
gets routed around in a spreadsheet, where the override is invisible and unmeasurable.

The literature and practice both show overrides are a mixed bag: small adjustments tend
to destroy value, large adjustments informed by genuine private information tend to add
it (Fildes et al. on judgemental adjustment). The distinction is only observable if
overrides are captured with their reason and their outcome.

Three options were considered. Block overrides above a threshold: pushes the work
off-system and destroys the measurement. Allow silently: keeps the workflow and leaves
us unable to tell good overrides from reflex. Allow with mandatory attribution: keeps
the workflow, costs the planner a few seconds, and produces the dataset needed to make
the trust argument later.

Overrides are also a signal, not just a cost. A cluster of overrides on one category is
often the earliest indication that an input broke.

## Decision

Overrides are never blocked. Every override records the planner, the magnitude, a reason
code from a short fixed list, and free text. Override rate and realised value-add are
reported per planner and per category monthly, to the planning organisation and not as
an individual performance metric. A category whose override rate rises sharply raises a
monitoring signal in the same channel as drift.

## Consequences

- Override behaviour becomes measurable, and the "does the forecast work" conversation
  can be evidence-based within a quarter.
- Planners pay a small friction cost per override and will resent it if the data is
  never shown back to them. Publishing the monthly view is a hard requirement, not a
  nice-to-have.
- Reporting per-planner numbers can be read as surveillance. Framing and audience are
  part of the decision: planning management, aggregated, never HR.
- Reason codes will be gamed toward whichever option is fastest to click. The free-text
  field is the mitigation, and the code list should be re-derived from the text annually.
- We accept that some value-destroying overrides continue to ship. The alternative
  destroys the ability to see them at all.
