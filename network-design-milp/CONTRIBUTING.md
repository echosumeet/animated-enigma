# Contributing

Bug reports and pull requests are welcome. This is a small library and the bar
is about evidence, not process.

## Getting set up

```bash
git clone https://github.com/echosumeet/network-design-milp
cd network-design-milp
python -m pip install -e ".[figures]"
make test
```

Runtime dependencies are `numpy` and `scipy` (the MILP solver is HiGHS, shipped
inside SciPy). `matplotlib` is only needed to regenerate the figures. Tests use
the standard library `unittest` and do not require `pytest`, though `pytest`
will collect them if you have it.

## The rules that matter

**1. A new model needs a test that checks the *solution*, not the objective.**

Asserting that the objective equals the number it returned yesterday tells you
something changed; it does not tell you what. Almost every real defect in a
network model is a modelling error, and modelling errors show up as violated
physics:

- flow conservation at every open facility,
- demand fully served,
- capacity and minimum-volume respected,
- no volume moving through a closed site.

`tests/test_network_flow.py` checks all four on every solve. Add to it.

**2. A new formulation needs its LP bound reported.**

Two formulations of the same problem have the same optimum and can have wildly
different relaxations. If you add or change a linking constraint, add a row to
the formulation table in `benchmarks/run_benchmarks.py` showing the LP bound
before and after. "It looked tighter" is not a measurement.

**3. Where a brute-force check is possible, do it.**

`facility_location.brute_force_uflp` enumerates every open set. It is the
reason the facility location code can be trusted. If your model has a small
instance whose optimum can be enumerated, enumerate it in a test.

**4. Every big-M is passed explicitly.**

`Model.link_big_m` has no default. If you find yourself wanting one, that is
the signal that you have not worked out the tight bound - and the tight bound
is usually `min(capacity, demand that could route here)`.

## Style

- Standard library `unittest`, no test framework dependency.
- Docstrings explain *why*. What the constraint says is visible in the code;
  why it is that constraint and not the other one usually is not.
- Keep model construction declarative. If a builder starts computing column
  offsets by hand, the modelling layer is missing a feature - add it there.
- No new runtime dependencies without a strong reason.

## Running everything

```bash
make all      # tests, benchmarks, figures, examples
```

CI runs the same on Python 3.10, 3.11 and 3.12.
