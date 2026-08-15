#!/usr/bin/env python3
"""The simulation engine on its own, without any supply-chain model on top.

Run: ``python examples/05_engine_tour.py``

``echelonsim.engine`` is a general discrete-event engine: a clock, a heap keyed
on ``(time, priority, insertion order)``, generator-based processes, and
interrupts. The supply-chain model is one thing you can build on it. This
example builds a different one -- a small cross-dock -- to show the engine's
contract directly:

1. events at the same timestamp resolve by priority, then by insertion order,
2. processes are ordinary Python generators that yield events,
3. one process can wait on another,
4. a process can be interrupted mid-wait and handle it.

The priority mechanism is the part that matters for modelling. Almost every
"the numbers are slightly off" bug in a periodic-review simulation is really a
disagreement about what happens first inside a single period, and an engine that
hides that ordering makes the disagreement impossible to see.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from echelonsim.engine import HIGH, LOW, NORMAL, URGENT, Environment, Interrupt

TRUCK_ARRIVALS = [(0.0, 40), (2.5, 25), (6.0, 60), (9.0, 30)]


def unloading_dock(env, state):
    """Consumes arrived pallets at a fixed rate, one period at a time."""
    while True:
        yield env.timeout(1.0, priority=NORMAL)
        moved = min(state["waiting"], state["rate"])
        state["waiting"] -= moved
        state["moved"] += moved
        if moved:
            state["cleared_at"] = env.now
            print(f"  t={env.now:5.1f}  dock moved {moved:3d} pallets, "
                  f"{state['waiting']:3d} still waiting")


def arrivals(env, state):
    for time, pallets in TRUCK_ARRIVALS:
        yield env.timeout(max(0.0, time - env.now), priority=HIGH)
        state["waiting"] += pallets
        print(f"  t={env.now:5.1f}  truck arrives with {pallets} pallets")
    return state["moved"]


def shift_supervisor(env, dock_process):
    """Halts the dock for a two-period safety stand-down."""
    yield env.timeout(4.0, priority=URGENT)
    print(f"  t={env.now:5.1f}  safety stand-down called")
    dock_process.interrupt(env.now + 2.0)


def dock_with_standdown(env, state):
    delay = 1.0
    while True:
        try:
            yield env.timeout(delay, priority=NORMAL)
        except Interrupt as interrupt:
            release = float(interrupt.cause)
            print(f"  t={env.now:5.1f}  dock stops")
            while env.now < release:
                yield env.timeout(1.0, priority=NORMAL)
            print(f"  t={env.now:5.1f}  dock restarts")
            delay = 1.0
            continue
        moved = min(state["waiting"], state["rate"])
        state["waiting"] -= moved
        state["moved"] += moved
        if moved:
            state["cleared_at"] = env.now
            print(f"  t={env.now:5.1f}  dock moved {moved:3d} pallets, "
                  f"{state['waiting']:3d} still waiting")
        delay = 1.0


def show_priority_ordering():
    print("1. Same timestamp, different priorities -- the order is the model.")
    env = Environment()
    for priority, label in ((LOW, "record statistics"),
                            (URGENT, "disruption starts"),
                            (NORMAL, "place orders"),
                            (HIGH, "receive shipments")):
        env.timeout(3.0, priority=priority).add_callback(
            lambda _event, text=label: print(f"     {text}")
        )
    env.run()
    print("   All four were scheduled for t=3.0. Priority decided the sequence,")
    print("   and insertion order would have decided any remaining ties.\n")


def show_processes():
    print("2. Generator processes, and one process waiting on another.")
    env = Environment()
    state = {"waiting": 0, "moved": 0, "rate": 20, "cleared_at": 0.0}
    env.process(unloading_dock(env, state), "dock")
    feed = env.process(arrivals(env, state), "arrivals")

    def closer():
        total = yield feed
        print(f"  t={env.now:5.1f}  last truck logged; {total} pallets moved "
              f"so far, {state['waiting']} still on the yard")

    env.process(closer(), "closer")
    env.run(until=14.0)
    print(f"   moved {state['moved']} pallets, yard cleared at "
          f"t={state['cleared_at']:.1f}\n")
    return state["cleared_at"]


def show_interrupt():
    print("3. An interrupt thrown into a running process.")
    env = Environment()
    state = {"waiting": 0, "moved": 0, "rate": 20, "cleared_at": 0.0}
    dock = env.process(dock_with_standdown(env, state), "dock")
    env.process(arrivals(env, state), "arrivals")
    env.process(shift_supervisor(env, dock), "supervisor")
    env.run(until=14.0)
    print(f"   moved {state['moved']} pallets, yard cleared at "
          f"t={state['cleared_at']:.1f}")
    return state["cleared_at"]


def main() -> int:
    show_priority_ordering()
    undisrupted = show_processes()
    disrupted = show_interrupt()
    print(f"   The same 155 pallets moved either way, but the yard cleared at "
          f"t={disrupted:.1f} instead of t={undisrupted:.1f}: the stand-down "
          f"was absorbed by\n   slack in the dock rate, and the only trace it "
          f"leaves is the delay. That is what makes\n   disruption recovery "
          f"worth simulating rather than reasoning about -- whether an "
          f"interruption\n   shows up downstream at all depends on slack that "
          f"is nowhere in the incident report.\n")
    print("About 300 lines of engine, no third-party simulation dependency, "
          "and a total order on\nevents that makes runs byte-identical across "
          "machines -- which is what common random\nnumbers need in order to "
          "work at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
