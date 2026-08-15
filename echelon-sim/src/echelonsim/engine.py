"""A discrete-event simulation engine built on ``heapq``.

This is deliberately hand-rolled rather than imported. A supply-chain simulation
lives or dies on two things that a general-purpose DES package hides from you:

1. **Tie-breaking.** Within a single simulated period a dozen things happen at
   the same timestamp -- trucks arrive, customers demand, nodes allocate, orders
   are placed, statistics are recorded. Which of those happens "first" is a
   modelling decision with real consequences (a receipt that lands before the
   allocation step is available to ship this period; one that lands after is
   not). Here that ordering is an explicit ``priority`` field on every event, so
   the period structure is visible in the code instead of emerging from the
   library's internals.

2. **Reproducibility.** The event heap is keyed on
   ``(time, priority, insertion_counter)``. The counter is a strictly increasing
   integer, so the ordering is a total order and two runs with the same seeds
   replay identically -- which is the precondition for common random numbers
   (see :mod:`echelonsim.rng`).

The programming model is the familiar generator-as-process one: a process is a
Python generator that yields events, and is resumed when the event it yielded is
processed. ``Process`` is itself an ``Event``, so one process can wait on
another. Interrupts are supported because supply disruptions are naturally
modelled as something happening *to* a running process.

Complexity is ``O(log n)`` per event, ``n`` being the number of pending events.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Any, Callable, Generator, List, Optional

__all__ = [
    "Environment",
    "Event",
    "Timeout",
    "Process",
    "Interrupt",
    "EngineError",
    "URGENT",
    "HIGH",
    "NORMAL",
    "LOW",
]

# --------------------------------------------------------------------------
# Priority bands. Lower numbers are processed first at a given timestamp.
# The simulation model in echelonsim.simulation lays its period phases out
# inside the NORMAL band; URGENT is reserved for interrupts and control
# signals that must pre-empt anything already queued for the same instant.
# --------------------------------------------------------------------------
URGENT = -100.0
HIGH = -10.0
NORMAL = 0.0
LOW = 100.0

_PENDING = object()


class EngineError(RuntimeError):
    """Raised on misuse of the engine (double-trigger, yielding a non-event)."""


class Interrupt(Exception):
    """Thrown into a process that another party interrupted."""

    @property
    def cause(self) -> Any:
        return self.args[0] if self.args else None


class Event:
    """Something that will happen, possibly now, possibly later.

    An event starts untriggered. ``succeed`` (or ``fail``) schedules it on the
    environment's heap; when the environment pops it, every registered callback
    runs with the event as its argument. Callbacks are the entire extension
    mechanism -- processes are implemented as a callback that resumes a
    generator.
    """

    __slots__ = ("env", "name", "callbacks", "value", "_scheduled", "_processed", "_failed", "_defused")

    def __init__(self, env: "Environment", name: str = "event") -> None:
        self.env = env
        self.name = name
        self.callbacks: Optional[List[Callable[["Event"], None]]] = []
        self.value: Any = _PENDING
        self._scheduled = False
        self._processed = False
        self._failed = False
        self._defused = False

    # -- state ----------------------------------------------------------
    @property
    def triggered(self) -> bool:
        """True once the event has been scheduled (it may not have fired yet)."""
        return self._scheduled

    @property
    def processed(self) -> bool:
        """True once the environment has popped the event and run callbacks."""
        return self._processed

    @property
    def failed(self) -> bool:
        return self._failed

    # -- triggering -----------------------------------------------------
    def succeed(self, value: Any = None, delay: float = 0.0, priority: float = NORMAL) -> "Event":
        if self._scheduled:
            raise EngineError(f"event {self.name!r} has already been triggered")
        self.value = value
        self.env.schedule(self, delay, priority)
        return self

    def fail(self, exception: BaseException, delay: float = 0.0, priority: float = URGENT) -> "Event":
        if self._scheduled:
            raise EngineError(f"event {self.name!r} has already been triggered")
        if not isinstance(exception, BaseException):
            raise EngineError("fail() needs an exception instance")
        self.value = exception
        self._failed = True
        self.env.schedule(self, delay, priority)
        return self

    def add_callback(self, callback: Callable[["Event"], None]) -> None:
        if self._processed:
            raise EngineError("cannot add a callback to an already-processed event")
        assert self.callbacks is not None
        self.callbacks.append(callback)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "processed" if self._processed else ("scheduled" if self._scheduled else "pending")
        return f"<{type(self).__name__} {self.name!r} {state}>"


class Timeout(Event):
    """An event that fires ``delay`` time units from now."""

    __slots__ = ("delay",)

    def __init__(self, env: "Environment", delay: float, value: Any = None,
                 priority: float = NORMAL, name: str = "timeout") -> None:
        if delay < 0:
            raise EngineError("negative timeout delay")
        super().__init__(env, name)
        self.delay = float(delay)
        self.value = value
        env.schedule(self, delay, priority)


class Process(Event):
    """A generator driven by the event loop.

    The generator yields events; the process resumes when each yielded event is
    processed, receiving that event's value. When the generator returns, the
    process event itself succeeds with the return value, so other processes can
    ``yield`` a process and wait for it to finish.
    """

    __slots__ = ("_generator", "_target")

    def __init__(self, env: "Environment", generator: Generator, name: str = "process") -> None:
        super().__init__(env, name)
        if not hasattr(generator, "send"):
            raise EngineError("Process needs a generator")
        self._generator = generator
        self._target: Optional[Event] = None
        boot = Event(env, f"{name}:start")
        boot.add_callback(self._resume)
        boot.succeed(priority=NORMAL)

    @property
    def is_alive(self) -> bool:
        return not self._processed

    def interrupt(self, cause: Any = None) -> None:
        """Throw an :class:`Interrupt` into the process at the current time."""
        if self._processed:
            raise EngineError("cannot interrupt a finished process")
        target = self._target
        if target is not None and target.callbacks is not None:
            try:
                target.callbacks.remove(self._resume)
            except ValueError:
                pass
        self._target = None
        signal = Event(self.env, f"{self.name}:interrupt")
        signal.add_callback(self._resume)
        signal.fail(Interrupt(cause), priority=URGENT)

    # -- internals ------------------------------------------------------
    def _resume(self, event: Event) -> None:
        self.env._active_process = self
        try:
            while True:
                try:
                    if event._failed:
                        event._defused = True
                        target = self._generator.throw(event.value)
                    else:
                        target = self._generator.send(event.value)
                except StopIteration as stop:
                    self._finish(stop.value)
                    return
                if not isinstance(target, Event):
                    raise EngineError(
                        f"process {self.name!r} yielded {target!r}, which is not an Event"
                    )
                if target._processed:
                    # Already fired (e.g. a zero-delay timeout consumed inside
                    # the same callback sweep) -- loop rather than deadlock.
                    event = target
                    continue
                target.add_callback(self._resume)
                self._target = target
                return
        finally:
            self.env._active_process = None

    def _finish(self, value: Any) -> None:
        self._target = None
        if not self._scheduled:
            self.value = value
            self.env.schedule(self, 0.0, HIGH)

    def _fail(self, exception: BaseException) -> None:
        """The generator raised. Re-raise it out of ``Environment.step``.

        A process that dies silently is the worst failure mode a simulation can
        have: the run completes, the numbers look plausible, and one echelon
        simply stopped ordering halfway through. The exception is attached to
        the process event and re-raised when nothing defuses it.
        """
        self._target = None
        if not self._scheduled:
            self.value = exception
            self._failed = True
            self.env.schedule(self, 0.0, HIGH)


class Environment:
    """The event loop: a clock, a heap, and a deterministic total order."""

    def __init__(self, initial_time: float = 0.0) -> None:
        self._now = float(initial_time)
        self._heap: List[tuple] = []
        self._counter = itertools.count()
        self._active_process: Optional[Process] = None
        self.events_processed = 0

    # -- clock ----------------------------------------------------------
    @property
    def now(self) -> float:
        return self._now

    @property
    def active_process(self) -> Optional[Process]:
        return self._active_process

    def __len__(self) -> int:
        return len(self._heap)

    # -- construction helpers -------------------------------------------
    def event(self, name: str = "event") -> Event:
        return Event(self, name)

    def timeout(self, delay: float, value: Any = None, priority: float = NORMAL,
                name: str = "timeout") -> Timeout:
        return Timeout(self, delay, value, priority, name)

    def process(self, generator: Generator, name: str = "process") -> Process:
        return Process(self, generator, name)

    def schedule(self, event: Event, delay: float = 0.0, priority: float = NORMAL) -> None:
        if delay < 0:
            raise EngineError("cannot schedule an event in the past")
        if event._scheduled:
            raise EngineError(f"event {event.name!r} is already scheduled")
        event._scheduled = True
        heapq.heappush(self._heap, (self._now + delay, priority, next(self._counter), event))

    def at(self, time: float, callback: Callable[[Event], None], priority: float = NORMAL,
           name: str = "at") -> Event:
        """Convenience: run ``callback`` at an absolute time."""
        event = Event(self, name)
        event.add_callback(callback)
        self.schedule(event, time - self._now, priority)
        return event

    # -- running --------------------------------------------------------
    def peek(self) -> float:
        return self._heap[0][0] if self._heap else math.inf

    def step(self) -> None:
        """Process exactly one event."""
        if not self._heap:
            raise EngineError("event queue is empty")
        time, _priority, _count, event = heapq.heappop(self._heap)
        self._now = time
        event._processed = True
        self.events_processed += 1
        callbacks, event.callbacks = event.callbacks, None
        for callback in callbacks or ():
            callback(event)
        if event._failed and not event._defused:
            raise event.value

    def run(self, until: Optional[float] = None) -> float:
        """Run until the queue empties or the clock reaches ``until``.

        ``until`` is exclusive: events scheduled exactly at ``until`` do not
        run. That makes ``run(until=T)`` mean "simulate periods 0..T-1", which
        is the convention the rest of the package assumes.
        """
        if until is None:
            while self._heap:
                self.step()
            return self._now
        if until < self._now:
            raise EngineError("cannot run backwards")
        while self._heap and self._heap[0][0] < until:
            self.step()
        self._now = float(until)
        return self._now
