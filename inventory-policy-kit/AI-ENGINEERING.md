# AI-ENGINEERING.md — inventory-policy-kit

## Overview

`inventory-policy-kit` is a classical inventory theory library: safety stock from loss
functions, four replenishment policies, lot sizing, newsvendor, Clark-Scarf decomposition,
Graves-Willems placement and risk pooling. Runtime dependencies are numpy and scipy. There
is no LLM and no learned model in the running product.

Against Andrew Ng's four-skill AI Engineering map, it demonstrates **software engineering
fundamentals**, **using coding agents** and **shaping the build**, but **not** building and
deploying AI applications — there is no model to build or deploy. What transfers is the
measurement discipline: build an oracle independent of the thing under test, then hold the
method to it.

## Disclosure

The code here was written by Claude subagents on 2026-08-15, in a single session, under
human direction (Sumeet, GitHub `echosumeet`). The human set the goal and constraints,
approved the portfolio composition and called a mid-run scope cut; the machine wrote the
design, implementation, tests, benchmark harness, figures and README. This document maps
both the product and the build process against the skills map. The suite is **121 tests**;
re-running `PYTHONPATH=src python -m unittest discover -s tests` while writing this, all 121
passed in 23.5 s.

## The four skills, as they actually appeared

### 1. Building and deploying AI applications — not present

No LLM, no learned model, no inference path. The transferable part is the evaluation
harness: **every analytic formula is checked against a Monte Carlo simulation of the policy
that formula produces.**

`src/invkit/simulation.py` is a hand-written `heapq`-based discrete-event evaluator. It
imports `policies` only for the `Policy` protocol and never a sizing function, so the check
is not circular. Tests in `tests/test_simulation_validation.py` build a policy from a
closed-form target and assert the delivered service matches, to 0.010 points on cycle
service and 0.004 on fill rate. From `benchmarks/results.md`:

| policy / measure | target | simulated | error |
|---|---|---|---|
| (s,Q) cycle service | 0.950 | 0.9519 | +0.0019 |
| (s,Q) fill rate | 0.950 | 0.9505 | +0.0005 |
| (R,S) R=1 ready rate | 0.950 | 0.9491 | -0.0009 |
| (R,S) R=5 fill rate | 0.980 | 0.9801 | +0.0001 |

Three exact oracles sit alongside it:

- `test_matches_brute_force_on_random_instances` (`tests/test_lotsizing.py`) checks the
  `O(T²)` Wagner-Whitin DP against exhaustive enumeration on **30 random instances** to 6
  decimal places.
- `test_dp_matches_brute_force_on_random_trees` (`tests/test_multiechelon.py`) checks the
  Graves-Willems tree DP against `enumerate_optimal_cost`, which brute-forces every integer
  service-time vector, on **20 random trees**, requiring exact agreement.
- Clark-Scarf DP cost against simulation of the policy it produces: **-0.45% / -0.45% /
  -0.33%** for 2-, 3- and 4-stage systems.

The negative controls matter more. Two tests assert *failure*:
`test_ignoring_undershoot_misses_the_target_low` pins the uncorrected textbook formula below
0.90 against a 0.95 target (measured 0.7932, 15.7 points short), and
`test_overlapping_orders_break_the_single_cycle_formula` pins the validity boundary — when
the lot is smaller than maximum lead-time demand, realised service drops to 0.8125 against a
0.90 target. That is an eval set containing cases the system is expected to fail, so a later
"improvement" that changes the failure mode gets caught.

### 2. Software engineering fundamentals

The load-bearing decision is in `docs/design.md`: **the loss function `E[(D-x)⁺]` is the
primitive, not the z-table.** Cycle service, fill rate, expected backorders and the
newsvendor objective are all loss-function evaluations, written once against a distribution
interface in `src/invkit/distributions.py`, so gamma, normal, empirical and mixture demand
get the same formulas. A `z` lookup with normality baked in is what makes most planning
systems structurally unable to answer "what if the errors are not normal?" At a 5-day lead
time and a 0.98 target, empirical quantile sizing asks for 197 units of safety stock against
the normal fit's 138 — **+42.1%** on identical aggregated moments.

Declining a dependency is also a fundamentals decision. `scipy.optimize.milp` was available
and not used (design.md, Decision 8): fill-rate inversion is a monotone scalar root find,
Wagner-Whitin a shortest path on a DAG, Clark-Scarf a chain of 1-D minimisations,
Graves-Willems a tree DP — 9 stages in 3 ms. The cost is stated rather than hidden:
fill-rate inversion runs **5.339 s per 1,000 solves**, which the README calls out as
unusable for a million-SKU nightly run and the first thing to fix.

### 3. Using coding agents

The agent-facing artifact was a shared `CONVENTIONS.md` at the portfolio root, handed to
every subagent as the spec and context contract: content rules, environment constraints,
required layout, README specification, definition of done. It, not the individual prompts,
is why ten independently-written repositories look like one portfolio.

The container had no PyPI access — only the preinstalled scientific Python core. Diagnosed
before any code was written and turned into a design rule: the hand-rolled `heapq` simulator
instead of simpy, and stdlib `unittest` tests, so the whole suite could be run and verified
inside the session.

The loop was closed by verifiers, not assertion. The definition of done required each agent
to actually run `python -m unittest discover -s tests`, `python benchmarks/run_benchmarks.py`
and its example scripts, and to write the README from the real output; fabricating a
benchmark number was prohibited. Every table in `README.md` is copied from
`benchmarks/results.md` (generation runtime 32.9 s). A separate audit agent that wrote none
of the code re-ran every suite and benchmark across all ten repositories and spot-checked
README numbers; it found one honesty defect portfolio-wide, in `eta-risk-engine`, and
corrected the claim to the measured value.

### 4. Shaping the build

Constraints were set up front: no employer reference, no scraped or copyrighted material,
MIT licence, all data self-generated in code — here, gamma demand with mean 100, sd 30,
5-day lead time, lot 800.

Two shaping decisions are visible in the repo. Gamma rather than normal, because gamma is
exactly closed under convolution at fixed scale, so a formula-versus-simulation disagreement
is a real defect rather than an approximation artifact — the model choice exists to make the
eval credible. And the framing is a business argument, not a formula catalogue: reading a
"95% service level" parameter as cycle service rather than fill rate costs **149 units** of
extra safety stock at the reference lot, **3,732 of working capital per SKU** at unit cost
25, delivering a fill rate of 0.9976 nobody asked for.

One portfolio-level decision the human made and lost: the initial request was to back-date
commit history across several years to disguise a gap in GitHub activity. Claude declined —
GitHub records repository creation and push events separately from author dates, making the
fabrication detectable and a worse signal than an empty graph. The counter-proposal was
accepted; this repository ships with one commit, dated 2026-08-15.

## Build narrative for this repository

| step | what happened | skill |
|---|---|---|
| Spec | `CONVENTIONS.md` handed to one subagent owning the repo end to end | shaping |
| Design | `docs/design.md`: loss function as primitive, gamma default, receive → order → demand timing, no MILP | SE fundamentals |
| Implementation | 14 modules, 3,534 lines in `src/invkit/` | coding agents |
| Verification | 121 tests, 1,285 lines in `tests/`; `benchmarks/run_benchmarks.py` → `benchmarks/results.md` | evals |
| Correction | portfolio-wide: a 1,200–2,500 line target overshot at ~6,000 lines/repo on two cores; the run was stopped after four repos, six relaunched under a 900–1,400 budget | shaping |
| Audit | independent agent re-ran suite and benchmarks, checked one commit, no backdating | coding agents |
| Commit | one commit, `Sumeet <benstokesipl@gmail.com>`, 2026-08-15 | — |

`src` + `tests` here total 4,819 lines, well above the budget imposed on the relaunched six;
`BUILD-NARRATIVE.md` does not name which four repos finished first, so I cannot say which
side of the cut this one fell on.

## AI during development vs AI in the product

| AI during development | AI in the product |
|---|---|
| Claude subagent wrote design, 14 modules, 121 tests, benchmark harness, four figures, README | *Empty. There is no AI in this product.* |
| Orchestrator fanned out one agent per repo, ~2 concurrent on 2 cores; `CONVENTIONS.md` was the shared context contract | |
| Independent audit agent re-ran every suite and benchmark | |

The second column is empty, in those words: loss functions, renewal theory, dynamic
programming, Monte Carlo. Nothing in it learns.

## What I would do differently

1. **Vectorise the fill-rate inversion.** 5.339 s per 1,000 solves is fine for analysis and
   unusable for a catalogue run; the bisection is trivially vectorisable across items.
2. **Fix lost sales rather than only documenting it.**
   `test_lost_sales_over_buffers_when_run_on_backorder_parameters` demonstrates the error
   without correcting the recursion.
3. **Give the guaranteed-service model a fat-tail escape.** Demand over `t` periods is
   assumed never to exceed `μt + zσ√t`; the failure mode is a zero-safety-stock pass-through
   stage — `build`, `pack`, `raw_A` in the BOM — turning out to be the constraint.
4. **Budget the agent by artifact, not by line count.** "One exact oracle per solver" would
   have protected the right thing; the mid-run line budget was a poor proxy for cost.

## Takeaways

1. **(Skill 1, transferred)** An eval is only worth running if the oracle shares no code with
   the thing under test. `simulation.py` deliberately does not import a sizing function; that
   restriction is what makes the ±0.002 agreements evidence rather than tautology.
2. **(Skill 1, transferred)** Negative tests carry more information than positive ones — the
   two failure assertions, 0.7932 without undershoot and 0.8125 with overlapping orders, pin
   the validity region, which is the part a user needs.
3. **(Skill 2)** The choice of primitive determines what the system can ever answer.
   `E[(D-x)⁺]` over a z-table is why empirical and mixture demand cost nothing extra, and why
   the +42.1% normal-assumption gap is measurable at all.
4. **(Skill 3)** Agents close their own loop only when the definition of done names the
   commands; every table in the README traces to `benchmarks/results.md` because of it.
5. **(Skill 4)** The plan was wrong about execution cost by roughly a factor of four; the
   right response was cutting scope, not extending the run.
## How to explore this repo

1. `tests/test_simulation_validation.py` — the falsification harness and the two tests that
   assert failure.
2. `docs/design.md` — eight decisions with their rejected alternatives.
3. `src/invkit/distributions.py`, then `src/invkit/safety_stock.py` — the primitive and what
   is built on it.
4. `benchmarks/results.md` — every number quoted in the README, with its DGP.
5. `src/invkit/guaranteed_service.py` — the tree DP and its `enumerate_optimal_cost` oracle.
