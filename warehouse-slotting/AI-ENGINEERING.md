# AI Engineering skills — `warehouse-slotting`

A slotting and pick-path optimisation library for a manual forward pick area: a
block-layout warehouse graph with an exact shortest-path metric, five slotting policies,
six routing policies held against two independent exact references, and two batching
heuristics. There is no AI in it — no LLM, no learned model, no run-time inference. It
is classical operations research over a distance metric. Against Andrew Ng's four-skill
map it demonstrates skills 2, 3 and 4 directly; skill 1 appears only in its transferable
half, the measurement discipline rather than the model.

## Disclosure

The code here was written by Claude subagents on 2026-08-15, in one session, under
human direction. The human set the goal, the constraints and the portfolio composition
and cut scope mid-run; the machine wrote every line of Python, ran every test and
produced every number. This document maps the product and the build process against the
skills map. The suite is 36 `unittest` cases and they pass:
`PYTHONPATH=src python -m unittest discover -s tests` reports `Ran 36 tests ... OK`.

## The headline result is a negative one

| Slotting policy | 2-opt travel (km, held out) | vs unslotted |
| --- | ---: | ---: |
| as-received (random feasible) | 181.4 | baseline |
| ABC by pick frequency | 85.5 | -52.9% |
| ABC + steepest descent | 86.3 | -52.4% |
| affinity (greedy clusters) | 88.1 | -51.4% |
| ABC + simulated annealing | 93.1 | -48.7% |
| affinity (spectral clusters) | 140.9 | -22.3% |

Plain ABC by pick frequency wins. Local search over a calibrated objective lands 1.0%
*worse* on travel; simulated annealing is worse still. The second finding is sharper:
`src/slotting/calibration.py` prices the affinity term at exactly **0.000** here. At
3.50 lines per order there is not enough co-occurrence mass for pairing to beat
proximity-to-depot, even though the generator deliberately puts a measured 5.07x
within-family lift into the stream. A tool that hard-codes a positive affinity weight
because affinity is fashionable will slot to it anyway.

Neither result was tuned away. The local-search loss is explained, not hidden: the
search is buying ergonomics with metres, lifting golden-zone share from 30.8% to 36.7%
at a rate the calibrated objective was told to accept (`src/slotting/ergonomics.py`).
Measuring only metres would call the search broken. The trade is legible only because
vertical cost was priced in metre-equivalents rather than folded into distance.

## Skill 1 — Building and deploying AI applications

No LLM and no learned model is in the running product. What transfers is the evaluation
harness — the same discipline an AI system needs, applied where ground truth is
available:

- **Held-out scoring.** Slotting sees the first 2,400 orders; every travel number is
  scored on the held-out 1,600 (`src/slotting/benchmark.py`). In-sample the winning
  policy reads 49.6 m per order against 53.4 m held out — the benefit decays **7.7%**.
  A train/test split applied to an optimiser, catching the ML failure mode exactly:
  report the fit, call it the benefit.
- **Exact references, not relative claims.** Routing heuristics are reported as an
  optimality gap against the Ratliff & Rosenthal aisle dynamic program, with Held-Karp
  run independently on every tour small enough for it and asserted to agree to 1e-6
  (`src/slotting/routing.py`, `tests/test_routing.py:75`). 2-opt: 0.08% mean gap.
  S-shape: 48.71%.
- **Error analysis that changed a conclusion.** The gap table was computed under both
  slotted and unslotted assignments, and the ranking inverts: S-shape goes from 16.56%
  to 48.71% mean gap, return routing from 19.72% down to 3.26%. That inversion exists
  only because the harness ran the comparison both ways.
- **Calibration at the margin.** `calibration.py` fits surrogate weights differentially
  around a reference assignment rather than on levels of travel, reporting R² 0.989 on
  move deltas. Fitting on levels gives a well-fitting, useless model: a swap decision
  needs the marginal rate, not the level.

## Skill 2 — Software engineering fundamentals

The central tradeoff is metric correctness against search speed. Euclidean distance is
wrong non-uniformly — two faces across a rack are 2.5 m apart in the plane and sixty
metres apart for a picker — so `src/slotting/layout.py` implements the closed-form
shortest path over the corridor graph, and `tests/test_layout.py` checks it pair-by-pair
against Dijkstra on an explicit `networkx` graph in one-block and two-block layouts. The
closed form is kept because local search calls it millions of times; the graph exists
only as a test oracle. Other choices:

- No PyPI access in the build container: Held-Karp, the aisle DP, Clarke-Wright savings
  and simulated annealing are written from the published formulations, not imported.
- Single-block heuristics raise on multi-block layouts rather than silently returning a
  wrong number (`tests/test_routing.py:123`). Failing loudly is the cheaper bug.
- Constraints are enforced structurally, not by post-hoc filtering: hazmat to floor
  level, weight capacity falling with level, cube fit (`tests/test_slotting.py:70`,
  `:80`, `:91`). Line 136 asserts the incremental swap delta matches a full objective
  recompute — the invariant the local search rests on.

## Skill 3 — Using coding agents

This repository was written by a subagent working from a shared `CONVENTIONS.md`
context contract — content rules, environment constraints, required layout, README
specification, definition of done — rather than a bespoke prompt. The contract, not the
prompt, is what made ten separately-written repositories read as one portfolio.

The loop was closed by verifiers: the definition of done required the agent to actually
run the test suite, `benchmarks/run_benchmarks.py` (11.7 s) and
`examples/slot_and_route.py`, and to write the README from real output. Fabricating a
benchmark number was prohibited outright.

The pitfall this repo hit is overrun. Against a 1,200–2,500 line target the first
repositories came in near 6,000 lines each, and on two effective cores the ten-repo run
projected to roughly three hours. The run was stopped after four repositories.
`warehouse-slotting` already had ~4,700 lines on disk, so rather than restart it under
the tightened budget the agent was redirected to finish what existed. It now stands at
5,841 lines of Python — over budget, and the honest record of a plan that was wrong
about cost. A separate audit agent that wrote none of the code re-ran every suite and
benchmark across all ten repositories and spot-checked README numbers against
`benchmarks/results.md`.

## Skill 4 — Shaping the build

- **Report the negative result.** The obvious move once local search lost by 1.0% was
  to tune the annealing schedule until it won. Instead the loss ships, with the
  golden-zone number that explains it. A slotting tool that always reports the
  sophisticated method winning is not a useful tool.
- **Exact references were scoped in deliberately.** Routing is the one part of this
  problem where optimal is cheap, so a heuristic without its gap was not acceptable.
- **Batching kept, congestion cut.** Batching under real cart limits takes the best
  slotting from 75.2 to 25.2 m per order — a larger gain than slotting itself, so it
  earned its place. Congestion was documented as a limitation, not faked.
- **The commit history is honestly dated.** The human initially asked for commits to be
  back-dated across several years to disguise a gap in GitHub activity. Claude declined:
  GitHub records repository creation and push events separately from author dates, so
  the fabrication is detectable and a worse signal than an empty graph. The
  counter-proposal was accepted.

## Build narrative for this repository

| Step | Skill | Decision point |
| --- | --- | --- |
| Spec via `CONVENTIONS.md` | 3 | Context contract shared across ten repos |
| Design (`docs/design.md`) | 2, 4 | Aisle-graph metric; ergonomics in metre-equivalents |
| Implementation (17 modules) | 2 | Closed form for speed, Dijkstra kept as test oracle |
| Verification | 1, 3 | 36 tests, held-out scoring, exact references agreeing to 1e-6 |
| Correction | 3, 4 | Overrun caught at ~4,700 lines; finish rather than restart |
| Audit | 1, 3 | Independent agent re-ran suite and benchmarks |
| Commit | 4 | One commit, honest author and committer date |

## AI during development vs AI in the product

| AI during development | AI in the product |
| --- | --- |
| A Claude subagent wrote all `src/`, `tests/`, `benchmarks/`, `examples/`, `docs/` | Empty. There is no AI in this product. |
| Design proposed by the agent, constraints set by the human | No LLM, no learned model, no run-time inference |
| An independent audit agent re-ran tests and benchmarks | Every result is deterministic given a seed |

The second column is empty. This is a classical operations research library.

## What I would do differently

1. **Budget the line count as a verifier, not a target.** At 5,841 lines this repo is
   over the stated band by more than a factor of two; the ceiling belonged in the
   definition of done.
2. **Calibrate affinity on a second instance before concluding.** The 0.000 weight is a
   statement about a 3.50-line basket, not about affinity slotting in general. A
   basket-size sweep would say where affinity starts to pay.
3. **Sweep the ergonomics weight.** The travel loss is one point on a trade curve;
   reporting travel and golden-zone share across a range would let a reader choose.
4. **Layer congestion over the tours.** Blocking rises with the concentration slotting
   creates, so -52.9% is an upper bound the model cannot currently bound.

## Takeaways

- **Skill 1.** A held-out split catches optimiser overfitting as it catches model
  overfitting: fitting and scoring on the same demand would have reported 49.6 m per
  order instead of the honest 53.4.
- **Skill 2.** Deciding once where correctness matters (the metric) and where speed
  matters (the inner loop), then keeping the slow oracle purely as a test, beats
  optimising both.
- **Skill 3.** Agents overrun when a budget is a target and hold when it is a verifier.
  This repo is the evidence for the first half of that sentence.
- **Skill 4.** Shipping the negative result — local search 1.0% worse, affinity weight
  0.000 — is what makes the rest of the numbers worth trusting.

## How to explore this repo

1. `benchmarks/results.md` — every headline number, regenerated in 11.7 s.
2. `docs/design.md` — why the metric and the held-out split are built as they are.
3. `src/slotting/calibration.py` — the differential fit that prices affinity at 0.000.
4. `src/slotting/routing.py` — the aisle DP and Held-Karp, cross-asserted in tests.
