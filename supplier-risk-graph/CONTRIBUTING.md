# Contributing

Small repository, simple rules.

- Tests are stdlib `unittest`. Run `make test` (or
  `PYTHONPATH=src python -m unittest discover -s tests -v`) before opening a PR.
- Every number in the README comes from `benchmarks/run_benchmarks.py`. If your
  change moves a number, rerun the benchmarks and commit the regenerated
  `benchmarks/results.md` and `docs/network_spof.png`.
- Dependencies are numpy, networkx, pydantic and matplotlib. Adding one needs a
  reason in the PR description.
- There is one availability kernel, `flow.output_fraction`. New analyses call it;
  they do not reimplement it. A second copy of that logic is how the structural
  view and the simulation start disagreeing.
- New modelling code needs a test that would fail without it. `tests/toy.py` is a
  hand-built network with a known answer — extend it rather than asserting against
  whatever the generator happens to produce.
