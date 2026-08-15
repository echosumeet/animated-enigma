# AI Engineering skills map — `network-design-milp`

A supply chain network design library: a four-echelon, multi-commodity, multi-mode
mixed-integer programme (`suppliers -> plants -> DCs -> zones`) on a modelling layer over
`scipy.optimize.milp` (HiGHS), plus a two-stage stochastic version reporting VSS and EVPI.
No LLM and no learned model in the running product. Classical operations research.

Against Andrew Ng's four AI Engineering skills (published 14 Aug 2026), this repository
demonstrates **software engineering fundamentals**, **using coding agents** and **shaping
the build**. It does **not** demonstrate building and deploying AI applications, because
there is no AI application here.

## Disclosure

The code was written by Claude subagents on 2026-08-15, in a single session, under human
direction (Sumeet, GitHub `echosumeet`). The human set the goal and constraints, chose the
portfolio composition, made the scope calls and rejected one proposal outright; the machine
designed, implemented, tested, benchmarked and documented the repository. Nothing here
should be read as claiming a human typed the code. The suite is **88 tests, stdlib
`unittest`, all passing** — re-verified with
`PYTHONPATH=src python -m unittest discover -s tests` (6.6s). None of this has run in
production or at scale.

## 1. Building and deploying AI applications — absent, with one transfer

No LLM, no learned model, no inference at runtime. Claiming otherwise would be
manufacturing an angle.
The transferable part is the evaluation harness, which exists for the reason an AI system
needs evals: output is plausible whether or not it is correct. A MILP with an off-by-one in
a block offset solves cleanly and answers a different question, with no exception and a
believable objective value. So it is checked against ground truth:

| check | where | compared against |
|:---|:---|:---|
| enumeration of all 255 open sets | `tests/test_facility_location.py` | exact optimum 439,554 |
| strong vs aggregated UFLP linking | `benchmarks/run_benchmarks.py` | LP bound 439,554 (gap 0.00%) vs 307,601 (30.02%) |
| `WS <= RP <= EEV` | `tests/test_stochastic.py` | theory, asserted on every solve |
| degenerate scenario set | same file | VSS and EVPI must both collapse to 0 |
| elastic relaxation vs priced shortfall | `examples/04_infeasibility_triage.py` | 61,271.6 units violation vs 61,271 unserved |

`docs/design.md` records that 87 of the 88 assertions check a property — conservation,
capacity, a theoretical inequality, an independently computed value — not a recorded number.
A regression test on an objective value says something changed; these say what broke.

## 2. Software engineering fundamentals

**Build a layer or call the solver directly.** `src/netdesign/modeling.py` (749 lines) adds
four things over the dense vectors `scipy.optimize.milp` expects — a variable registry keyed
by tuples, wildcard selection over a positional inverted index, sparse expression algebra,
COO/CSR assembly — and deliberately no presolve, cuts or column generation, because HiGHS
does that better.

**Big-M as an API decision.** `Model.link_big_m` (`src/netdesign/modeling.py:498`) has no
default for `M`. The cost of getting it wrong — same feasible set, same optimum of 1,917,707
in every row:

| formulation | rows | LP bound | integrality gap | MILP time |
|:---|---:|---:|---:|---:|
| disaggregated link, tight M | 626 | 1,773,829 | 7.50% | 0.44s |
| aggregate link, tight M | 100 | 1,722,375 | 10.19% | 0.16s |
| aggregate link, M x100 | 112 | 1,019,910 | 46.82% | 0.15s |

`benchmarks/results.md` also records the caveat that undercuts the point: HiGHS presolve
tightens coefficients and recovers most of the wall-clock loss at these sizes.

**Environment constraint as design rule.** The container had no PyPI access, so the solver
is driven through `scipy.optimize.milp` rather than pulp or ortools, tests are stdlib
`unittest`, and `figures.py` is imported only by the figure script so matplotlib is never on
the test path. Scaling is measured: the default instance is 690 vars in 0.41s, the largest
4,543 vars and 18,398 nonzeros in 3.74s.

## 3. Using coding agents

One subagent owned this repository end to end — design, code, tests, benchmarks, figures,
README — under an orchestrator fanning out one agent per repository across ten. The container
had 2 cores, so effective concurrency was about two agents. Three mechanisms did the
steering:

- **A shared context contract.** One `CONVENTIONS.md` carried the content rules,
  environment constraints, required layout, README specification and definition of done to
  every subagent — that file, not the individual prompts, is why ten independently written
  repositories read as one portfolio.
- **Verifiers, so the agent closes its own loop.** Done required actually running
  `python -m unittest discover -s tests -v`, `python benchmarks/run_benchmarks.py` and the
  examples, and writing the README from real output. `benchmarks/results.md` opens with
  "Every table in the README is copied from this file."
- **An independent audit.** A separate agent that wrote none of the code re-ran every suite
  and benchmark across all ten repositories and checked README numbers against
  `benchmarks/results.md`. It found one honesty defect in the portfolio — `eta-risk-engine`
  claimed an 11-second pipeline against a measured 4.2 seconds — and corrected it. Nothing
  was found here.

## 4. Shaping the build

The product decision here is what to measure. The optimum opens three of eight DCs at
1,917,707 per period; the best two-DC network is +1.36%, the best four-DC network +1.56%.
The cost curve is flat across the whole region anyone would argue about, and no demand
forecast is accurate to 1.5%. Shipping a single open set would have been defensible and
misleading. So the deliverables became the numbers that survive that flatness:

- **VSS = 58,313 per period (2.57% of RP)** — what modelling the uncertainty is worth.
- **EVPI = 251,576 per period (11.10% of RP)** — a ceiling on any forecast-accuracy
  investment; most such business cases are pitched above it.
- **A partition instead of an answer.** Across a 15-cell demand x transport-rate sweep
  producing 5 distinct networks, `DC03` and `DC05` are core (open in >=90% of runs),
  `DC00`, `DC01`, `DC04`, `DC07` are swing, none is never-open.

Ownership shows in the choices that keep those numbers honest. Recourse is forced on
(`solve_two_stage` sets `allow_unmet=True`) because without it `EEV = +inf` and VSS is
undefined. Single sourcing sits in the first stage because it is a commercial arrangement,
not a routing decision. The sampler is three-level because independent per-zone noise
diversifies away across 25 zones and returns VSS of exactly zero.

## Build narrative

Spec (`CONVENTIONS.md`) → environment diagnosis, no PyPI, before any code → layering fixed
in `docs/design.md` → the decision to make VSS/EVPI and core/swing/out the deliverable → 14
modules, 5,053 lines → 88 tests, benchmarks and five examples run → one commit, `Sumeet
<benstokesipl@gmail.com>`, 2026-08-15 07:16:36 +0000.

**The overrun.** Against a 1,200–2,500 line target the early repositories came in near
6,000 lines each, and on two effective cores the full run projected to roughly three hours.
This repository is one of the four already committed when the run was stopped — 5,053
lines, roughly double the target. The remaining six were relaunched under a hard budget of
900–1,400 lines, 20–35 tests, one figure, one example, with an explicit protect-list. The
plan was wrong about cost; the correction cut scope on work not yet started.

**The rejected proposal.** The human initially asked for commit history back-dated across
several years to disguise a gap in GitHub activity. Claude declined: GitHub records
repository creation and push events separately from author dates, so the fabrication is both
detectable and a worse signal than an empty graph. `git log` shows one commit with matching
author and committer dates.

## AI during development vs AI in the product

| AI during development | AI in the running product |
|:---|:---|
| Claude subagents wrote the design, code, tests, benchmarks, figures and docs | None. No LLM, no learned model. |
| `CONVENTIONS.md` as context contract across ten parallel agents | Optimisation is exact: HiGHS. |
| Tests and benchmarks as the agent's verifier | `greenfield.py` is Weiszfeld (1937). |

The right-hand column is empty, and that is the accurate description of this repository.

## What I would do differently

- **Put inventory in the objective.** DC-count decisions are driven as much by safety stock,
  which grows roughly with the square root of stocking locations, as by transport. A
  piecewise-linear approximation over the open count fits the MILP directly.
- **Treat 12 scenarios as a demonstration, not a measurement.** Enough to show VSS and EVPI,
  far too few to trust 58,313 and 251,576 as magnitudes. Sample average approximation with
  confidence intervals on the gap is the next step.
- **Budget the repository before writing it.** 5,053 lines against a 1,200–2,500 target is a
  planning failure the verifier loop cannot catch: passing tests say nothing about scope.
- **Make service a constraint, not a metric.** The optimum puts the average unit 883 km from
  its DC, which no consumer-facing network accepts.

## Takeaways

1. *Shaping the build.* When the top three answers sit within 1.6%, the optimum is not the
   deliverable — the core/swing/out partition and the EVPI ceiling are. Falsifiable: had the
   sweep produced 15 distinct networks rather than 5, the partition would be worthless.
2. *Software engineering fundamentals.* API design is bound quality. Removing the default
   from `link_big_m` is the difference between a 7.50% and a 46.82% integrality gap on the
   same feasible set.
3. *Using coding agents.* Verifiers beat instructions. "Do not fabricate numbers" is
   unenforceable; "copy the README tables from the benchmark output" is checkable, and an
   audit agent checked it.
4. *Using coding agents.* Agents miss cost, not correctness — every test passed, and the
   failure was a 2x scope overrun.

## How to explore this repo

1. `README.md` — the VSS/EVPI headline and every measured table.
2. `docs/design.md` — why the modelling layer exists and why big-M has no default.
3. `src/netdesign/modeling.py` — registry, wildcard selection, sparse assembly (`link_big_m`,
   line 498).
4. `src/netdesign/stochastic.py`, `tests/test_stochastic.py`, `benchmarks/results.md`.
