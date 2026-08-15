# AI Engineering — supply-chain-copilot

A tool-calling planning copilot over a synthetic warehouse star schema: a guarded text-to-SQL
layer, five typed tools, a small agentic loop, and a 25-question eval harness with a second
paraphrase arm. It runs end to end with no API key and no network.

Mapped against Andrew Ng's AI Engineering Skills Map (14 Aug 2026), this is the one repository
in the portfolio where all four skills appear in the product and not only in the build. Skill 1
is the point of it; skill 2 is the layering that makes the eval meaningful; skills 3 and 4 show
in how it was built and in what was protected when the build ran over budget.

## Disclosure

The code here was written by Claude subagents on 2026-08-15, in a single session, under human
direction (Sumeet, GitHub `echosumeet`). A human set the goal, constraints and scope; the
machine designed, implemented, tested and benchmarked. Nothing was hand-written by a person and
presented as agent output, and none of it has ever run in production, at a company, or at any
scale beyond this container. 35 tests, all passing (`PYTHONPATH=src python -m unittest discover
-s tests`); the portfolio total is 678. This document maps both the product and the build
process against the four skills.

## The four skills

### 1. Building and deploying AI applications

**Agentic loop.** `src/sccopilot/agent.py` is 107 lines: plan, call, observe, respond, with the
operationally interesting parts explicit. `MAX_STEPS = 6` is a hard budget; exhausting it
produces a stated refusal (`stopped_on_budget`), not a guess. A `ToolError` goes back into the
message list as an observation and the loop continues rather than crashing. Every step emits a
trace record with step index, tool name, arguments and latency; observations are truncated at
320 characters by `_summarise`, because a real trace store cannot hold every row a tool
returned.

**Context engineering.** The model never sees the database. `guard.schema_prompt()` renders a
compact schema card from `ALLOWED_SCHEMA` — seven tables, their readable columns, one line of
join keys — into `SYSTEM_PROMPT`. Card and authorizer read the same dict, so what the model
believes exists and what it may read cannot drift apart. `internal_credentials` exists in the
database and appears in neither.

**Typed tool registry.** `src/sccopilot/tools.py` exposes `sql_query`, `inventory_position`,
`stockout_risk`, `supplier_lead_time_stats` and `run_what_if` with pydantic argument models
published as JSON Schema. A `ValidationError` becomes a `ToolError`, so a malformed call is a
recoverable observation, not a stack trace.

**Guarded text-to-SQL.** Four layers in `src/sccopilot/guard.py`: static string checks,
`sqlite3.set_authorizer`, `LIMIT` injection with clamping (`DEFAULT_LIMIT = 100`,
`MAX_ROWS = 200`), a read-only handle. All nine attacks in `evals.ATTACKS` are blocked:

| attack | blocked by |
| --- | --- |
| drop table, stacked statement, attach database | static check: stacked statements |
| delete rows, pragma probe | static check: SELECT/WITH only |
| comment smuggling | static check: comment tokens |
| union exfiltration, schema enumeration, blocked column | authorizer: outside allowlist |

The split is the finding. The six loud attacks die at the regex with a readable refusal. The
three a keyword denylist would pass — a `UNION` onto an unlisted table, `SELECT name FROM
sqlite_master`, a direct read of `internal_credentials.key_value` — are stopped inside the
parser, where the authorizer is consulted per table, column and function at prepare time.

**Evals.** `src/sccopilot/evals.py` holds 25 golden questions across seven categories (lookup,
aggregation, multi-hop, what-if, ambiguous, refusal, adversarial), three graders (contains,
numeric with relative tolerance, tool-trajectory match), and a paraphrase arm restating all 25
intents with the same truth and trajectory. Numeric ground truth is an independent SQL
statement run outside the tool layer, so drifting tool arithmetic fails the eval rather than
agreeing with itself.

| arm | pass rate | tool-selection accuracy | avg steps |
| --- | --- | --- | --- |
| golden phrasings | 1.00 | 1.00 | 1.92 |
| paraphrased | 0.48 | 0.72 | 1.64 |

Per category the paraphrase arm reads: ambiguous 1.00, adversarial 1.00, lookup 0.60,
multi-hop 0.50, what-if 0.33, refusal 0.33, aggregation 0.00 — 13 of 25 fail. The 1.00 is a
harness check, not a model score: the stub is a rule-based router, so it only says the tools,
truth queries and graders agree. The 0.48 carries the information, and its shape is legible —
aggregation collapses to 0.00 because those questions route on SQL templates keyed to specific
noun phrases, while ambiguity and adversarial handling survive rephrasing because they trigger
on structure. The router has not been tuned against the paraphrase arm; that would move the
brittleness somewhere the eval cannot see.

**The stub as a testability decision.** `StubProvider` in `src/sccopilot/providers.py` is the
default: deterministic, offline, pattern-matched, and the reason the evals and attack suite run
in CI with no API key and no spend. `AnthropicProvider` and `OpenAIProvider` are drop-in and
import their SDKs lazily. Nothing above `providers.py` knows which model is answering; nothing
below `tools.py` knows a model exists.

### 2. Software engineering fundamentals

Six modules, one dependency direction (`docs/design.md`). That layering is what lets a stub run
and a hosted run exercise the same tools, guard and graders — without it the eval measures a
different system than the one that ships. `LIMIT` injection is filed as a cost and
context-window concern, not a security one. Tool-selection accuracy is reported apart from
answer accuracy: a right answer via three tools is a cost problem, a right tool with a wrong
filter is a correctness problem. The warehouse is generated by code — 7,261 rows, seed
20260815, 180 days, 40 SKUs, 6 locations, 8 suppliers, built in 73 ms — so the repo is
reproducible from a clone. Tests are stdlib `unittest`, including
`test_warehouse_is_not_mutated_by_attacks` and `test_guard_leaves_no_authorizer_installed`.

### 3. Using coding agents

One subagent per repository, fanned out by an orchestrator against a shared `CONVENTIONS.md`
carrying the content rules, layout, README spec and definition of done. That definition was a
verifier, not an assertion: the agent had to run its tests, benchmark and example and write the
README from the real output; fabricating a number was prohibited. Re-running
`benchmarks/run_benchmarks.py` while writing this document reproduced every accuracy figure
exactly (latency figures move with the host).

### 4. Shaping the build

Two decisions shaped what this repository is. When the build ran over budget, the guardrail
suite and the eval harness were named as the things that could not be cut and the HTTP serving
layer went instead — a copilot without evals is a demo. Earlier, the human asked for back-dated
commits to disguise a gap in GitHub activity; that was declined, because GitHub records
creation and push events separately from author dates, making the fabrication both detectable
and a worse signal than an empty graph.

## Build narrative

1. **Spec** (skill 4). `CONVENTIONS.md` plus a brief: a planning copilot with real guardrails
   and a real eval, the eval harness named as the deliverable.
2. **Design** (skill 2). Six modules, one dependency direction, provider protocol at the
   boundary — what made stub and hosted model interchangeable.
3. **Implementation.** Warehouse, guard, tools, agent, evals, CLI: 1,342 lines of `src`.
4. **Verification** (skill 3). 35 tests, the 9-attack suite and both eval arms, run in-session
   against runnable verifiers rather than asserted.
5. **Correction** (skill 4). The first run overshot — near 6,000 lines against a 1,200–2,500
   target, ~3 hours projected on two cores. Stopped after four repositories, relaunched under a
   900–1,400 line, 20–35 test budget with a protected list.
6. **Audit and commit.** A separate agent that wrote none of the code re-ran every test and
   benchmark across the ten repositories. One commit, honest date.

## AI during development vs AI in the product

| AI during development | AI in the running product |
| --- | --- |
| Subagents wrote all code, tests, benchmarks, figures | The agent loop in `agent.py`, behind an `LLMProvider` |
| A shared conventions file as context contract | A schema card generated from the allowlist |
| Verifiers: run tests and benchmarks, write from output | Evals: 25 questions, two arms, three graders |
| An audit agent re-ran what it had not written | The guardrail suite, run with the benchmark |

Both columns are populated here. In five of the ten portfolio repositories the right-hand
column is empty; this is not one of those.

## What I would do differently

1. Run the same 25 questions against a hosted model and report both side by side. Until then
   the harness is the deliverable and the 1.00 says nothing about model capability.
2. Replace the `Answer:` marker the numeric grader keys off with a structured output contract;
   a looser parser is how graders start passing things they should not.
3. Build a recovery taxonomy. There is one strategy today — drop the optional location filter,
   retry once — and bad arguments, empty results and timeouts each need a different response.
4. Add cost and token accounting and run-over-run regression comparison.

## Takeaways

- **Skill 1:** a single-arm eval measures the phrasings you wrote the system against. The
  paraphrase arm cost about thirty lines and turned 1.00 into 0.48.
- **Skill 2:** put the guard where the access happens. Six of nine attacks die at the regex;
  the three a denylist would have missed die at the authorizer. Reporting tool-selection
  accuracy separately from answer accuracy (1.00/1.00 golden, 0.72/0.48 paraphrased) is what
  localises that kind of fault.
- **Skill 3:** an agent given a runnable verifier closes its own loop; the one honesty defect
  the audit found portfolio-wide was in a README, not in a benchmark.
- **Skill 4:** the useful part of a budget overrun is naming what may not be cut before you
  start cutting.

## How to explore this repo

`src/sccopilot/guard.py` first — `ALLOWED_SCHEMA`, `_authorizer`, `static_check`. Then
`src/sccopilot/evals.py` (`GOLDEN_SET`, `PARAPHRASES`, `ATTACKS`), `src/sccopilot/agent.py` for
the loop, `docs/design.md` for the reasoning. `benchmarks/results.md` holds every published
number; `examples/planner_session.py` runs six questions offline.
