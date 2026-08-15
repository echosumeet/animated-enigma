# Contributing

Bug reports and pull requests are welcome. The bar here is evidence, not process.

## Getting set up

```bash
git clone https://github.com/echosumeet/echelon-sim
cd echelon-sim
python -m pip install -e ".[figures]"
make test
```

Runtime dependencies are `numpy` and `scipy`. `matplotlib` is only needed to
regenerate the figures in `docs/`. Tests use the standard library `unittest` and
do not require `pytest` (though `pytest` will collect them if you have it).
There is no simulation-framework dependency and there should not be one — the
engine in `src/echelonsim/engine.py` is part of the point of the project.

## The two rules that matter

**1. A simulator result needs an independent check.**

Simulators fail silently. A timing convention that is off by one period does not
raise; it produces plausible numbers that are wrong by 30%. So every claim about
the model needs something outside the model to check it against:

- a closed-form special case (`tests/test_bullwhip_analytics.py` runs the whole
  stack against the Chen, Drezner, Ryan & Simchi-Levi moving-average result and
  against the exponential-smoothing analogue derived in
  `src/echelonsim/bullwhip.py`),
- a conservation identity (`tests/test_network_and_simulation.py` checks that
  units are neither created nor destroyed, and that backlog equals cumulative
  demand less cumulative shipments, at every node),
- or a degenerate case that can be derived by hand (an oracle forecast with no
  batching must produce orders exactly equal to demand).

Asserting that a function returns the number it returned yesterday is not a test
of a simulation model.

**2. Anything that changes the period structure needs a test that pins it.**

The event priorities in `src/echelonsim/simulation.py` *are* the model. Moving
the receive phase after allocation, or the review phase before it, changes every
number the package produces and breaks nothing visibly. If you touch the phase
table, add a test that fails when the order changes.

## Style

- Standard library `unittest`, no test framework dependency.
- Docstrings explain *why*. The formula is usually obvious; the reason it is that
  formula and not the other one usually is not.
- Cite the literature for any published method, in the module docstring. If you
  derive something rather than cite it, say so and show the derivation — see the
  exponential-smoothing bullwhip expression for the pattern.
- Type hints on public functions.
- Keep the dependency direction downward through the layering in
  `docs/design.md`. In particular `simulation.py` must not import `bullwhip.py`;
  the simulator having no knowledge of the formulas it is validated against is
  what makes those tests meaningful.

## Before opening a pull request

```bash
make test       # every test passes
make bench      # benchmarks/results.md regenerates
make examples   # every example script runs
```

If a benchmark number changes, commit the regenerated `benchmarks/results.md`
alongside the change. Numbers in the README are copied from that file and from
nowhere else.

## Things I would particularly welcome

- **Lost sales** instead of backordering, and an honest statement of what
  changes. The fill-rate and recovery metrics both assume backorders.
- **The rationing game.** `Allocation.PROPORTIONAL` is implemented and creates
  the incentive to inflate orders; no node currently exploits it. A strategic
  ordering policy that does would complete Lee, Padmanabhan & Whang's four
  causes — three of the four are already measurable here.
- **Capacitated echelon policies.** `capacity` is enforced in allocation but no
  policy is aware of it, so a capacitated node under-orders in exactly the way
  real ones do. That is realistic, but a capacity-aware policy would be a useful
  comparison.
- **Correlated demand across retailers** in the divergent network, which is what
  makes real pooling benefits smaller than the square-root law predicts.
- **A sequential stopping rule** on replications: run until the relative
  half-width of the target metric is below a threshold, rather than a fixed count.
