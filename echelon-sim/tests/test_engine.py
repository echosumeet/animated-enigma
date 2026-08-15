"""Engine tests.

The engine's contract is a *total order* on events. Almost every subtle bug in a
supply-chain simulation is really a violation of that order, so these tests are
about ordering and determinism rather than about arithmetic.
"""

import unittest

from echelonsim.engine import (
    HIGH,
    LOW,
    NORMAL,
    URGENT,
    EngineError,
    Environment,
    Event,
    Interrupt,
    Process,
    Timeout,
)


class TestEventOrdering(unittest.TestCase):
    def test_events_fire_in_time_order(self):
        env = Environment()
        log = []
        for delay in (5.0, 1.0, 3.0, 0.5):
            env.timeout(delay).add_callback(lambda e, d=delay: log.append(d))
        env.run()
        self.assertEqual(log, [0.5, 1.0, 3.0, 5.0])

    def test_priority_breaks_time_ties(self):
        env = Environment()
        log = []
        for priority, label in ((LOW, "low"), (URGENT, "urgent"), (NORMAL, "normal"), (HIGH, "high")):
            env.timeout(2.0, priority=priority).add_callback(lambda e, l=label: log.append(l))
        env.run()
        self.assertEqual(log, ["urgent", "high", "normal", "low"])

    def test_insertion_order_breaks_priority_ties(self):
        env = Environment()
        log = []
        for index in range(6):
            env.timeout(1.0, priority=NORMAL).add_callback(lambda e, i=index: log.append(i))
        env.run()
        self.assertEqual(log, list(range(6)))

    def test_run_until_is_exclusive_and_advances_the_clock(self):
        env = Environment()
        log = []
        for delay in (1.0, 2.0, 3.0):
            env.timeout(delay).add_callback(lambda e, d=delay: log.append(d))
        env.run(until=3.0)
        self.assertEqual(log, [1.0, 2.0])
        self.assertEqual(env.now, 3.0)
        # The event at exactly 3.0 is still pending and runs on continuation.
        env.run()
        self.assertEqual(log, [1.0, 2.0, 3.0])

    def test_zero_delay_events_still_advance_through_the_queue(self):
        env = Environment()
        log = []
        first = env.timeout(0.0)
        first.add_callback(lambda e: log.append("first"))
        second = env.timeout(0.0)
        second.add_callback(lambda e: log.append("second"))
        env.run()
        self.assertEqual(log, ["first", "second"])
        self.assertEqual(env.now, 0.0)


class TestErrors(unittest.TestCase):
    def test_negative_delay_rejected(self):
        env = Environment()
        with self.assertRaises(EngineError):
            env.timeout(-1.0)

    def test_double_trigger_rejected(self):
        env = Environment()
        event = Event(env)
        event.succeed(1)
        with self.assertRaises(EngineError):
            event.succeed(2)

    def test_callback_on_processed_event_rejected(self):
        env = Environment()
        event = Event(env)
        event.succeed()
        env.run()
        with self.assertRaises(EngineError):
            event.add_callback(lambda e: None)

    def test_stepping_an_empty_queue_rejected(self):
        env = Environment()
        with self.assertRaises(EngineError):
            env.step()

    def test_yielding_a_non_event_is_an_engine_error(self):
        env = Environment()

        def bad():
            yield 42

        env.process(bad())
        with self.assertRaises(EngineError):
            env.run()


class TestProcesses(unittest.TestCase):
    def test_process_resumes_after_each_timeout(self):
        env = Environment()
        stamps = []

        def ticker():
            for _ in range(4):
                yield env.timeout(2.5)
                stamps.append(env.now)

        env.process(ticker())
        env.run()
        self.assertEqual(stamps, [2.5, 5.0, 7.5, 10.0])

    def test_process_receives_the_event_value(self):
        env = Environment()
        seen = []

        def consumer():
            value = yield env.timeout(1.0, value="payload")
            seen.append(value)

        env.process(consumer())
        env.run()
        self.assertEqual(seen, ["payload"])

    def test_one_process_can_wait_on_another(self):
        env = Environment()
        order = []

        def worker():
            yield env.timeout(4.0)
            order.append(("worker done", env.now))
            return "result"

        def waiter():
            child = env.process(worker())
            value = yield child
            order.append(("waiter resumed", env.now, value))

        env.process(waiter())
        env.run()
        self.assertEqual(order[0], ("worker done", 4.0))
        self.assertEqual(order[1], ("waiter resumed", 4.0, "result"))

    def test_process_exception_propagates_out_of_run(self):
        env = Environment()

        def explodes():
            yield env.timeout(1.0)
            raise ValueError("boom")

        env.process(explodes())
        with self.assertRaises(ValueError):
            env.run()

    def test_interrupt_is_thrown_into_the_waiting_process(self):
        env = Environment()
        log = []

        def sleeper():
            try:
                yield env.timeout(100.0)
                log.append("slept through")
            except Interrupt as interrupt:
                log.append(("interrupted", env.now, interrupt.cause))
                yield env.timeout(1.0)
                log.append(("resumed", env.now))

        process = env.process(sleeper())

        def interrupter():
            yield env.timeout(3.0)
            process.interrupt("outage")

        env.process(interrupter())
        env.run()
        self.assertEqual(log[0], ("interrupted", 3.0, "outage"))
        self.assertEqual(log[1], ("resumed", 4.0))

    def test_interrupt_cancels_the_original_wakeup(self):
        """The pre-empted timeout must not also resume the process later."""
        env = Environment()
        resumes = []

        def sleeper():
            try:
                yield env.timeout(10.0)
            except Interrupt:
                pass
            resumes.append(env.now)
            yield env.timeout(50.0)
            resumes.append(env.now)

        process = env.process(sleeper())

        def interrupter():
            yield env.timeout(2.0)
            process.interrupt(None)

        env.process(interrupter())
        env.run()
        self.assertEqual(resumes, [2.0, 52.0])

    def test_interrupting_a_finished_process_is_an_error(self):
        env = Environment()

        def quick():
            yield env.timeout(1.0)

        process = env.process(quick())
        env.run()
        with self.assertRaises(EngineError):
            process.interrupt()


class TestDeterminism(unittest.TestCase):
    def test_identical_programs_produce_identical_traces(self):
        def build():
            env = Environment()
            trace = []

            def agent(name, period):
                for _ in range(20):
                    yield env.timeout(period, priority=NORMAL)
                    trace.append((env.now, name))

            for name, period in (("a", 1.0), ("b", 1.0), ("c", 0.7)):
                env.process(agent(name, period))
            env.run(until=15.0)
            return trace

        self.assertEqual(build(), build())

    def test_event_counter_matches_processed_events(self):
        env = Environment()

        def agent():
            for _ in range(5):
                yield env.timeout(1.0)

        env.process(agent())
        env.run()
        # 5 timeouts + 1 boot event + 1 completion event for the process.
        self.assertEqual(env.events_processed, 7)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
