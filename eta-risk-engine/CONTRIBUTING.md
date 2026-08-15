# Contributing

Small repository, simple rules.

- Tests are stdlib `unittest`. Run `make test` (or
  `PYTHONPATH=src python -m unittest discover -s tests -v`) before opening a PR.
- Any number that appears in the README must come from
  `benchmarks/run_benchmarks.py`. If your change moves a number, rerun the
  benchmarks and commit the regenerated `benchmarks/results.md` and figure.
- Dependencies are numpy, pandas, scipy, scikit-learn and matplotlib. Adding one
  needs a reason in the PR description.
- New modelling code needs a test that would fail without it. Tests that assert
  the evaluation protocol (temporal splits, out-of-fold encoding) are load
  bearing — do not relax them to make a run go green.
- Keep the data generator and the models separate. Nothing in `src/etarisk`
  outside `generate.py` may read a `latent_*` column.
