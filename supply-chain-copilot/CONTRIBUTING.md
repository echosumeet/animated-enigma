# Contributing

Small repository, simple rules.

- Tests are stdlib `unittest`. Run `make test` (or
  `PYTHONPATH=src python -m unittest discover -s tests -v`) before opening a PR.
- Any number in the README must come from `benchmarks/run_benchmarks.py`. If your
  change moves a number, rerun the benchmarks and commit the regenerated
  `benchmarks/results.md` and `docs/eval_by_category.png`.
- Everything must keep running with no API key and no network. `StubProvider` is
  the default; the hosted providers import their SDK inside the method body and
  no test may touch that path.
- Guardrail tests are load bearing. Adding a table or column to
  `ALLOWED_SCHEMA` needs a reason in the PR description, and new attack cases are
  always welcome. Do not relax an attack test to make a run go green.
- Do not tune the router against the paraphrase arm of the eval. Fixing a
  paraphrase failure by adding the paraphrase's vocabulary to the router defeats
  the point of having the arm; add the capability instead, or leave the failure
  visible.
