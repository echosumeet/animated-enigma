# Contributing

Small repository, simple rules.

- Tests are stdlib `unittest`. Run `make test` (or
  `PYTHONPATH=src python -m unittest discover -s tests -v`) before opening a PR.
- Every number in the README comes from `benchmarks/run_benchmarks.py`. If your
  change moves a number, rerun the benchmarks and commit the regenerated
  `benchmarks/results.md` and `docs/stp_curve.png`.
- The default provider must stay offline and deterministic. Nothing under
  `src/tradedoc` may require an API key or a network call on an import path that
  tests or benchmarks touch. Hosted providers import their SDK inside the method.
- Nothing outside `generate.py` may read a ground-truth field. Extraction sees the
  rendered text and nothing else; if a test needs truth, it gets it from
  `GeneratedDoc.truth` after extraction has run.
- New extraction rules need a test that fails without them. The regression tests
  for blank-label capture and for numeric fields containing OCR letters are load
  bearing -- do not relax them to make a run go green.
- Dependencies are numpy, pydantic, matplotlib and reportlab. Adding one needs a
  reason in the PR description.
