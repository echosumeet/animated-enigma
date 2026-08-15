# Contributing

Bug reports and pull requests are welcome. This is a small library and the bar is
mostly about evidence, not process.

## Getting set up

```bash
git clone https://github.com/echosumeet/inventory-policy-kit
cd inventory-policy-kit
python -m pip install -e ".[figures]"
make test
```

Runtime dependencies are `numpy` and `scipy`. `matplotlib` and `networkx` are only
needed to regenerate the figures. Tests use the standard library `unittest` and do
not require `pytest` (though `pytest` will collect them if you have it).

## The one rule that matters

**A new analytic result needs a test that checks it against something
independent.** In practice that means one of:

- a Monte Carlo simulation of the policy the formula produces
  (`invkit.simulation.simulate_policy`),
- a brute-force enumeration of the same optimisation
  (`invkit.lotsizing` and `invkit.guaranteed_service` both ship one),
- or a closed-form special case that can be derived by hand.

Asserting that a function returns the number it returned yesterday is not a test
of an inventory formula. Almost every real defect in this domain is a modelling
error - a timing convention off by one period, a protection interval of `L`
instead of `R + L`, a missing undershoot term - and none of those are visible
without an independent check.

## Style

- Standard library `unittest`, no test framework dependency.
- Docstrings explain *why*, not *what*. The formula is usually obvious; the reason
  it is that formula and not the other one usually is not.
- Cite the literature for any published method, in the module docstring.
- Type hints on public functions.
- Keep the dependency direction downward through the layering in `docs/design.md`.
  In particular, `simulation.py` must not import a sizing function - that would
  make the validation tests circular.

## Before opening a pull request

```bash
make test       # all tests pass
make bench      # benchmarks/results.md regenerates
make examples   # every example script runs
```

If a benchmark number changes, commit the regenerated `benchmarks/results.md` with
the change. Numbers in the README are copied from that file and nowhere else.

## Things I would particularly welcome

- Lost-sales analogues of the sizing formulas.
- Intermittent-demand models (Croston, Syntetos-Boylan) feeding the same
  distribution interface.
- A vectorised fill-rate inversion that solves a whole catalogue at once.
- Capacitated variants of the multi-echelon models, with an honest statement of
  what breaks.
