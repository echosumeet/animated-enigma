# Contributing

Bug reports and pull requests are welcome.

## Running things

```bash
python -m pip install -e ".[figures]"
make test        # stdlib unittest, no pytest required
make bench       # regenerates benchmarks/results.md
make figures     # regenerates docs/pick_path.png
```

The suite is plain `unittest`, so it runs with no test dependencies at all. It
is also collected correctly by `pytest` if you happen to have it.

## What a good change looks like

- **Numbers in the README come from `benchmarks/run_benchmarks.py`.** If your
  change moves them, re-run it and commit the regenerated `results.md` and the
  README edits in the same commit. Do not hand-edit a benchmark number.
- **New routing heuristics need an exactness test.** Every router is checked
  against `exact_aisle_dp` and, where the instance is small enough, against
  `held_karp`. A heuristic that beats the exact route is a bug in the router or
  in the metric, and the suite should catch it before review does.
- **New geometry needs a Dijkstra test.** The closed-form travel metric is fast
  because it exploits the block layout. Any change to it must still match a
  shortest path on the explicit corridor graph (`Warehouse.to_networkx`) for
  every pair of locations in a small warehouse.
- Keep the dependency list as it is: numpy, scipy, scikit-learn, networkx, and
  matplotlib for figures only.

## Style

Type hints on public functions, docstrings that say why rather than what, and
no commented-out code. Line length 100.
