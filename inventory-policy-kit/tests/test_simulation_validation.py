"""Analytic formulas versus simulation. This is the part of the repo that matters.

Every test here builds a policy from a closed-form service target and then runs
the policy through the Monte Carlo evaluator to check that the service it
promised is the service it delivers.  A formula that has not been through this is
a formula nobody should be running a network on.

Tolerances are stated in service-level points and are deliberately tight (0.5-1.0
points).  They are achievable because the demand model is gamma - exactly closed
under convolution, exactly non-negative - so there is no truncation error to hide
behind, and because the reorder points include the undershoot correction.
"""

from __future__ import annotations

import unittest

from invkit.leadtime import LeadTimeSpec, ltd_with_undershoot
from invkit.policies import SQPolicy, build_RS, build_RsS, protection_interval_ltd
from invkit.safety_stock import ss_from_cycle_service_level, ss_from_fill_rate
from invkit.simulation import DemandProcess, simulate_policy, simulate_policy_replications

DEMAND_MEAN = 100.0
DEMAND_SD = 30.0
LEAD_TIME = 5
Q = 800.0

N_PERIODS = 120_000
WARMUP = 2_000


def _spec(lead_time=LEAD_TIME):
    return LeadTimeSpec.deterministic(DEMAND_MEAN, DEMAND_SD, lead_time)


def _process():
    return DemandProcess(DEMAND_MEAN, DEMAND_SD, "gamma")


class TestCycleServiceLevelIsAchieved(unittest.TestCase):
    """The analytic reorder point must deliver the cycle service level it claims."""

    def test_sQ_cycle_service_hits_target(self):
        proc = _process()
        ltd = ltd_with_undershoot(_spec(), proc.mean, proc.sd)
        for target in (0.90, 0.95, 0.98):
            res = ss_from_cycle_service_level(ltd, target, Q)
            policy = SQPolicy(s=res.reorder_point, Q=Q)
            sim = simulate_policy(
                policy, proc, {LEAD_TIME: 1.0}, n_periods=N_PERIODS, warmup=WARMUP, seed=101
            )
            self.assertAlmostEqual(
                sim.cycle_service_level, target, delta=0.010,
                msg=f"target {target}: simulated {sim.cycle_service_level:.4f}",
            )

    def test_ignoring_undershoot_misses_the_target_low(self):
        """The control: drop the undershoot term and service comes in short.

        This is the failure mode the correction exists to fix, and it is worth an
        explicit test because the uncorrected formula is what most systems run.
        """
        proc = _process()
        naive_ltd = protection_interval_ltd(_spec(), review_period=0, family="gamma")
        res = ss_from_cycle_service_level(naive_ltd, 0.95, Q)
        policy = SQPolicy(s=res.reorder_point, Q=Q)
        sim = simulate_policy(
            policy, proc, {LEAD_TIME: 1.0}, n_periods=N_PERIODS, warmup=WARMUP, seed=101
        )
        self.assertLess(sim.cycle_service_level, 0.90)

    @staticmethod
    def _bimodal_ltd(pmf, proc):
        from invkit.distributions import GammaLTD, MixtureLTD
        from invkit.leadtime import undershoot_moments

        eu, vu = undershoot_moments(proc.mean, proc.sd)
        components = [
            GammaLTD.from_moments(
                l * proc.mean + eu, (l * proc.sd ** 2 + vu) ** 0.5
            )
            for l in sorted(pmf)
        ]
        return MixtureLTD(components, [pmf[l] for l in sorted(pmf)])

    def test_stochastic_lead_time_cycle_service_hits_target(self):
        """The hard case: lead time is bimodal, so the mixture must be exact.

        The lot is sized above the maximum lead-time demand so that at most one
        order is outstanding at a time, which is the condition the single-cycle
        reorder-point formula actually assumes.  See the companion test for what
        happens when it is violated.
        """
        proc = _process()
        pmf = {3: 0.6, 9: 0.4}
        ltd = self._bimodal_ltd(pmf, proc)
        big_Q = 1500.0
        for target in (0.90, 0.95):
            res = ss_from_cycle_service_level(ltd, target, big_Q)
            policy = SQPolicy(s=res.reorder_point, Q=big_Q)
            sim = simulate_policy(
                policy, proc, pmf, n_periods=200_000, warmup=WARMUP, seed=202
            )
            self.assertAlmostEqual(
                sim.cycle_service_level, target, delta=0.010,
                msg=f"target {target}: simulated {sim.cycle_service_level:.4f}",
            )

    def test_overlapping_orders_break_the_single_cycle_formula(self):
        """Known limitation, pinned down rather than left as a caveat.

        When the lot covers less demand than the longest lead time, a second
        order is placed before the first lands.  Two orders are then exposed to
        the same demand, the cycles are no longer independent, and realised cycle
        service falls well below the formula - here by roughly nine points.  Any
        item with a long, variable lead time and a small lot is in this regime,
        and that combination is extremely common on low-volume imported parts.
        """
        proc = _process()
        pmf = {3: 0.6, 9: 0.4}
        ltd = self._bimodal_ltd(pmf, proc)
        small_Q = 800.0  # 8 periods of demand, shorter than the 9-period lead time
        res = ss_from_cycle_service_level(ltd, 0.90, small_Q)
        sim = simulate_policy(
            SQPolicy(s=res.reorder_point, Q=small_Q), proc, pmf,
            n_periods=200_000, warmup=WARMUP, seed=202,
        )
        self.assertLess(sim.cycle_service_level, 0.85)


class TestFillRateIsAchieved(unittest.TestCase):
    """The loss-function inversion must deliver the fill rate it claims."""

    def test_sQ_fill_rate_hits_target(self):
        proc = _process()
        ltd = ltd_with_undershoot(_spec(), proc.mean, proc.sd)
        for target in (0.95, 0.98, 0.995):
            res = ss_from_fill_rate(ltd, target, Q)
            policy = SQPolicy(s=res.reorder_point, Q=Q)
            sim = simulate_policy(
                policy, proc, {LEAD_TIME: 1.0}, n_periods=N_PERIODS, warmup=WARMUP, seed=303
            )
            self.assertAlmostEqual(
                sim.fill_rate, target, delta=0.004,
                msg=f"target {target}: simulated {sim.fill_rate:.5f}",
            )

    def test_fill_rate_target_holds_far_less_stock_than_csl_target(self):
        """Same number, read two ways, measured in simulation rather than assumed."""
        proc = _process()
        ltd = ltd_with_undershoot(_spec(), proc.mean, proc.sd)
        csl_res = ss_from_cycle_service_level(ltd, 0.95, Q)
        fill_res = ss_from_fill_rate(ltd, 0.95, Q)
        sims = {}
        for name, res in (("csl", csl_res), ("fill", fill_res)):
            sims[name] = simulate_policy(
                SQPolicy(s=res.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
                n_periods=N_PERIODS, warmup=WARMUP, seed=404,
            )
        self.assertGreater(sims["csl"].avg_on_hand, sims["fill"].avg_on_hand + 100.0)
        self.assertAlmostEqual(sims["fill"].fill_rate, 0.95, delta=0.004)
        self.assertGreater(sims["csl"].fill_rate, 0.995)


class TestPeriodicReview(unittest.TestCase):
    def test_RS_ready_rate_hits_target_with_R_equal_1(self):
        """With R = 1 every period is its own cycle, so CSL is the ready rate."""
        proc = _process()
        for target in (0.95, 0.98):
            policy, _ = build_RS(_spec(3), R=1, target_csl=target)
            sim = simulate_policy(
                policy, proc, {3: 1.0}, n_periods=N_PERIODS, warmup=WARMUP, seed=505
            )
            self.assertAlmostEqual(
                sim.ready_rate, target, delta=0.006,
                msg=f"target {target}: simulated {sim.ready_rate:.4f}",
            )

    def test_RS_fill_rate_formula_holds_for_a_weekly_review(self):
        proc = _process()
        for target in (0.95, 0.98):
            policy, res = build_RS(_spec(3), R=5, target_fill=target)
            sim = simulate_policy(
                policy, proc, {3: 1.0}, n_periods=N_PERIODS, warmup=WARMUP, seed=606
            )
            self.assertAlmostEqual(
                sim.fill_rate, target, delta=0.005,
                msg=f"target {target}: simulated {sim.fill_rate:.5f}",
            )

    def test_protection_interval_is_R_plus_L_not_L(self):
        """Sizing an (R, S) policy on L alone is a common and expensive error."""
        proc = _process()
        spec = _spec(3)
        correct, _ = build_RS(spec, R=5, target_csl=0.95)
        wrong_ltd = protection_interval_ltd(spec, review_period=0)
        wrong = type(correct)(R=5, S=ss_from_cycle_service_level(wrong_ltd, 0.95, 500.0).reorder_point)
        sim_wrong = simulate_policy(
            wrong, proc, {3: 1.0}, n_periods=N_PERIODS, warmup=WARMUP, seed=707
        )
        self.assertLess(sim_wrong.fill_rate, 0.75)
        self.assertGreater(correct.S, wrong.S)

    def test_RsS_suppresses_small_orders_versus_RS(self):
        proc = _process()
        spec = _spec(3)
        rs, _ = build_RS(spec, R=1, target_csl=0.95)
        rss, _ = build_RsS(spec, R=1, Q=800.0, target_csl=0.95)
        sim_rs = simulate_policy(rs, proc, {3: 1.0}, n_periods=40_000, warmup=WARMUP, seed=808)
        sim_rss = simulate_policy(rss, proc, {3: 1.0}, n_periods=40_000, warmup=WARMUP, seed=808)
        self.assertLess(sim_rss.n_orders, sim_rs.n_orders / 4)
        self.assertGreater(sim_rss.order_quantity_mean, sim_rs.order_quantity_mean * 4)


class TestSimulatorInternals(unittest.TestCase):
    def test_replications_report_a_confidence_half_width(self):
        proc = _process()
        ltd = ltd_with_undershoot(_spec(), proc.mean, proc.sd)
        res = ss_from_cycle_service_level(ltd, 0.95, Q)
        out = simulate_policy_replications(
            SQPolicy(s=res.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
            n_replications=6, n_periods=20_000,
        )
        mean, hw = out["cycle_service_level"]
        self.assertGreater(hw, 0.0)
        self.assertLess(abs(mean - 0.95), 0.02)

    def test_lost_sales_over_buffers_when_run_on_backorder_parameters(self):
        """Parameters derived under backordering are conservative for lost sales.

        Backordered demand is subtracted from the inventory position and then
        consumes stock when the replenishment lands, so the shortfall propagates
        into the next cycle.  Under lost sales the shortfall simply disappears.
        The same (s, Q) therefore holds *more* stock and serves a higher share of
        realised demand in a lost-sales world - which means a retailer running
        backorder-derived reorder points is quietly over-serving and over-holding.
        """
        proc = _process()
        ltd = ltd_with_undershoot(_spec(), proc.mean, proc.sd)
        res = ss_from_cycle_service_level(ltd, 0.90, Q)
        policy = SQPolicy(s=res.reorder_point, Q=Q)
        back = simulate_policy(policy, proc, {LEAD_TIME: 1.0}, n_periods=60_000, seed=909)
        lost = simulate_policy(
            policy, proc, {LEAD_TIME: 1.0}, n_periods=60_000, seed=909, lost_sales=True
        )
        self.assertGreater(lost.avg_on_hand, back.avg_on_hand)
        self.assertGreaterEqual(lost.fill_rate, back.fill_rate)

    def test_demand_process_rescaling_preserves_aggregate_moments(self):
        proc = _process()
        sliced = proc.rescaled(1.0 / 10)
        self.assertAlmostEqual(sliced.mean * 10, proc.mean, places=9)
        self.assertAlmostEqual(sliced.sd ** 2 * 10, proc.sd ** 2, places=9)

    def test_average_on_hand_matches_the_analytic_decomposition(self):
        from invkit.policies import average_inventory_sQ

        proc = _process()
        ltd = ltd_with_undershoot(_spec(), proc.mean, proc.sd)
        res = ss_from_cycle_service_level(ltd, 0.95, Q)
        decomposition = average_inventory_sQ(ltd, res.reorder_point, Q)
        sim = simulate_policy(
            SQPolicy(s=res.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
            n_periods=N_PERIODS, warmup=WARMUP, seed=111,
        )
        self.assertAlmostEqual(
            sim.avg_on_hand, decomposition["expected_on_hand"],
            delta=0.04 * decomposition["expected_on_hand"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
