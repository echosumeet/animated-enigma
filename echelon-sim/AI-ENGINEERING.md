# AI Engineering skills, mapped against `echelon-sim`

This repository is a discrete-event simulation of a multi-echelon supply chain on a
hand-written `heapq` event engine (`src/echelonsim/engine.py`, 330 lines), with the
output analysis done to simulation-methodology standards: common random numbers,
estimated warm-up truncation, batch-means confidence intervals, and validation against
closed-form results the simulator has no knowledge of.

It contains no AI — no LLM, no learned model, no inference call anywhere in the running
product. It is classical operations research and statistics. Against Andrew Ng's
four-skill map it demonstrates **software engineering fundamentals**, **using coding
agents** and **shaping the build** directly, and the *measurement discipline* underneath
**building AI applications** without the model-handling half of that skill at all.

## Disclosure

The code in this repository was written by Claude subagents on 2026-08-15, in a single
session, under human direction. The human set the goal, the constraints and the scope
and called the scope cut described below; the machine wrote the design notes, engine,
experiments, tests and benchmark harness. This document maps both the product and the
build process against the skills map. The suite is **143 tests**, all passing
(`PYTHONPATH=src python -m unittest discover -s tests`, 40.7s); the benchmark run takes
189s and writes `benchmarks/results.md`, from which every README number is copied.

## The four skills, as they actually appeared

### 1. Building and deploying AI applications — the eval half only

No LLM or learned model is in the running product. The transferable part is the
evaluation harness. It is strong here because the system is stochastic — the same code
and configuration return a different answer every run, which is the problem an AI eval
has to solve — but unlike an AI system, ground truth is sometimes available in closed
form. Three mechanisms (`src/echelonsim/metrics.py`, `experiments.py`):

| mechanism | defends against | measured effect |
|---|---|---|
| validation against closed form (`bullwhip.py`) | the whole stack silently computing the wrong thing | 6 configurations, worst error **+0.53%** (§1) |
| MSER-5 warm-up truncation (White 1997) | initialisation bias reported as a result | 6-period-transit cold start: **+49.9%** bias over a 720-period run, transient 140 periods (§7) |
| batch-means intervals (Schmeiser 1982) | correlated observations counted as independent | naive half-width **±22.7** units vs **±34.6**, 1.5× wider, same 660-period series (§7b) |

The closed-form check matters most: `benchmarks/run_benchmarks.py` runs the full engine,
network and event loop against Chen, Drezner, Ryan & Simchi-Levi's Theorem 1 and against
a smoothing analogue derived in `src/echelonsim/bullwhip.py`. Simulated 2.9160 against
2.9200, 7.0372 against 7.0000.

The ablation equivalent is section 3 of `benchmarks/results.md`: a full `2^3` factorial
over signal processing, batching and lead time, with Shapley attribution on log
amplification. Sequential ablation is order-dependent — lead time alone amplifies by
**1.00**, so a one-at-a-time ablation scores it zero and sends the reader to fix the
wrong mechanism. Under Shapley it takes +0.446 ± 0.010, entirely from interaction.

Genuinely absent: no prompt engineering, no context management, no RAG, no agentic
workflow, no handling of unpredictable model output.

### 2. Software engineering fundamentals

The central tradeoff is `docs/design.md` decision 1: hand-roll the event engine rather
than import a DES framework. The container had no PyPI access, but the recorded reason
is the modelling one — a dozen things happen at the same timestamp within a simulated
period, and which happens first changes the answers, so tie-breaking belongs in the
source rather than in a framework's internals. The heap is keyed on
`(time, priority, insertion_counter)`, a total order, which is the precondition for
byte-identical replay and therefore for common random numbers. Cost: 330 lines and 19
tests in `tests/test_engine.py`.

Two correctness decisions carry money:

- **Protection interval `R + L − 1`, not `R + L`** (`docs/design.md` decision 2). The
  textbook expression assumes review before the period's demand is served; here
  allocation runs at priority 30 and review at priority 50. Using `R + L` inflates
  every target by a full period of mean demand — 100 units on a 100-unit item, 400
  units of system inventory across four echelons. Pinned by
  `TestTimingConvention.test_oracle_base_stock_reproduces_demand_exactly` in
  `tests/test_network_and_simulation.py`, an exact identity any off-by-one breaks.
- **`on_order` and `in_transit` are separate fields** (decision 3). A real bug during
  the build, whose symptom was not a crash: a VMI configuration producing zero orders,
  0% fill rate and a plausible-looking inventory number. `TestEchelonAccounting` now
  pins each half of the rule separately.

The suite is built on invariants rather than golden values: `TestConservation` asserts
units are neither created nor destroyed under stochastic lead times and that backlog
equals cumulative demand less cumulative shipments.

### 3. Using coding agents

The context contract, not the prompt, did the work. A shared `CONVENTIONS.md` handed to
every subagent fixed the content rules, environment constraints, repository layout and
an explicit definition of done. That definition was a set of **verifiers the agent had
to actually run** — the unittest suite, `benchmarks/run_benchmarks.py`, each
`examples/*.py` — plus a rule that the README be written from the real output.
Fabricating a benchmark number was prohibited, and that is checkable: `results.md` is
generated and the README is copied from it. The pitfall this repo hit was cost
estimation, covered below. A separate audit agent, which had written none of this code,
later re-ran every suite and benchmark across all ten repositories and checked commit
authorship, layout and README numbers against `benchmarks/results.md`.

### 4. Shaping the build

The decisions here were about what to *measure*, not what to build.

- **Calibrate before comparing.** `src/echelonsim/information.py` bisects each mode's
  safety factor until retailer fill rate hits 97.5% before any inventory number is
  compared. Without that, "sharing POS data cuts inventory 63%" is partly a service cut.
- **Report the ranking that disagrees with the headline.** Shared POS wins on
  amplification (−94.7% ± 0.1) and inventory (−62.8% ± 0.2); VMI wins on cost
  (−42.7% ± 0.5). The README explains why rather than picking one column.
- **Pick the metric that does not flatter.** Variance ratio, not coefficient of
  variation: CV confounds amplification with a change in mean throughput, exactly what
  happens during a recovery (`docs/design.md` decision 5).

The repo therefore leads with what a planner cannot see from inside a node: local
amplification of 3.99, 5.01 and 4.29 compounding to **85.84 ± 1.87** at the factory, and
an 8-period outage whose 0.6% fill-rate trough arrives 13 periods in, five periods after
the supplier is back.

## Build narrative

| step | what happened | skill | decision point |
|---|---|---|---|
| spec | `CONVENTIONS.md`; no PyPI access diagnosed up front | 3 | make the constraint a design rule, not a workaround |
| design | `docs/design.md` written first, ten numbered decisions | 2, 4 | hand-roll the engine so tie-breaking is explicit |
| implementation | engine, network, policies, forecasters, four experiment modules | 2 | `R + L − 1`; splitting `on_order` from `in_transit` after the VMI zero-order bug |
| verification | 143 tests, benchmarks (189s), five examples, four figures, all run | 1, 3 | README written from `results.md`, not from memory |
| correction | overshoot: **7,181 lines of Python** against a 1,200–2,500 target | 3, 4 | the human stopped the run after four repos had committed and budgeted the remaining six at 900–1,400 lines |
| commit | one commit, `Sumeet <benstokesipl@gmail.com>`, honest date | 4 | back-dating was proposed and declined |

This repo was one of the four that committed before the stop, so the budget applied to
the six that followed, not to it. The overrun is visible in the artefact.

## AI during development vs AI in the product

| AI during development | AI in the running product |
|---|---|
| Claude subagents wrote the engine, network, policies, experiments, 143 tests, benchmarks, figures and README on 2026-08-15 | **None.** No LLM, no learned model, no inference call; the package imports `numpy` and `scipy` only. |
| `CONVENTIONS.md` served as spec and context contract | — |
| Verifiers (tests, benchmarks, examples) closed the agent's loop | — |
| A separate audit agent re-ran everything it had not written | — |

The second column is empty, and that is the accurate description of this repository.

## What I would do differently

1. **Budget the line count against a verifier, not a prompt.** 7,181 lines against a
   1,200–2,500 target went uncaught until the run was already long; a size check in the
   definition of done would have caught it in the first repository.
2. **Add a sequential stopping rule.** The replication count is fixed at 30, and the
   lead-time grid burns most of its runtime on cells that converged five iterations ago.
   Running until the relative half-width is under a threshold would spend the 189s where
   the variance actually is.
3. **Model a strategic orderer.** Proportional allocation creates the incentive to
   inflate orders, but every policy here orders its true requirement — so three of Lee,
   Padmanabhan & Whang's four bullwhip causes are measurable and the rationing game is
   not.
4. **Make capacity anticipated, not just enforced.** No policy knows its own capacity,
   so the capacity-loss recovery number (17 periods for an 8-period 50% loss) is worse
   than it needs to be.

## Takeaways

- **(Skill 1)** Ground truth beats plausibility: six closed-form checks with a worst
  error of 0.53% are worth more than any amount of "the output looks reasonable".
- **(Skill 1)** One-at-a-time ablation mis-ranks interacting components — lead time
  scores 1.00 alone and +0.446 under Shapley.
- **(Skill 2)** The expensive defects do not raise. The `R + L` off-by-one and the
  echelon double-count both produced fine-looking numbers; both are now pinned by
  exact-identity tests.
- **(Skill 3)** Verifiers, not instructions, make an agent's output trustworthy. "Run
  the benchmark and copy the number" is checkable; "be accurate" is not.
- **(Skill 4)** Equalising what you are not comparing is most of the work: every
  comparison is calibrated to a fixed fill rate first, and every disruption scenario is
  paired against its own undisrupted twin.

## How to explore this repo

1. `docs/design.md` — ten numbered decisions, including the two bugs above.
2. `src/echelonsim/engine.py`, then `examples/05_engine_tour.py`, which exercises it
   with no supply chain model attached.
3. `src/echelonsim/metrics.py` — MSER-5, batch means, paired-t, variance-ratio intervals.
4. `benchmarks/results.md` and `tests/test_network_and_simulation.py`
   (`TestConservation`, `TestEchelonAccounting`, `TestTimingConvention`).
