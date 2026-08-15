"""Network structure and the conservation laws the simulator must obey.

The most valuable tests here are the accounting identities. A multi-echelon
simulation has exactly two ways to be wrong that produce plausible-looking
output: it can lose units, and it can double-count pipeline. Both are checked
directly.
"""

import unittest

import numpy as np

from echelonsim.demand import IIDNormal
from echelonsim.forecast import ExponentialSmoothing, MovingAverage, Oracle
from echelonsim.leadtime import Deterministic, GammaLeadTime
from echelonsim.network import (
    Allocation,
    InfoMode,
    Node,
    SupplyNetwork,
    divergent_network,
    serial_chain,
)
from echelonsim.policies import BaseStock
from echelonsim.rng import StreamBank
from echelonsim.simulation import (
    CapacityLoss,
    DisruptionPlan,
    Simulator,
    SupplyOutage,
    run_simulation,
)


def build_chain(**kwargs):
    kwargs.setdefault("levels", 3)
    kwargs.setdefault("transit_factory", lambda level: Deterministic(2.0))
    return serial_chain(**kwargs)


class TestNetworkStructure(unittest.TestCase):
    def test_serial_chain_wires_suppliers_and_customers(self):
        network = build_chain()
        self.assertEqual(network.nodes["retailer"].supplier, "distributor")
        self.assertEqual(network.nodes["distributor"].customers, ["retailer"])
        self.assertTrue(network.source.is_source)
        self.assertEqual([n.name for n in network.retailers], ["retailer"])

    def test_divergent_network_pools_retailers_under_one_distributor(self):
        network = divergent_network(n_retailers=3)
        self.assertEqual(len(network.retailers), 3)
        self.assertEqual(len(network.nodes["distributor"].customers), 3)

    def test_missing_supplier_is_rejected(self):
        with self.assertRaises(ValueError):
            SupplyNetwork([Node("a", 0, supplier="nowhere")])

    def test_two_roots_are_rejected(self):
        nodes = [
            Node("a", 0, supplier=None, is_source=True),
            Node("b", 1, supplier=None, is_source=True),
        ]
        with self.assertRaises(ValueError):
            SupplyNetwork(nodes)

    def test_cycle_is_rejected(self):
        nodes = [
            Node("a", 0, supplier="b"),
            Node("b", 1, supplier="a"),
            Node("root", 2, supplier=None, is_source=True),
        ]
        with self.assertRaises(ValueError):
            SupplyNetwork(nodes)

    def test_zero_order_lead_time_is_rejected_with_an_explanation(self):
        with self.assertRaises(ValueError) as context:
            Node("a", 0, supplier="b", order_lead_time=0)
        self.assertIn("order_lead_time", str(context.exception))

    def test_protection_interval_is_review_plus_lead_minus_one(self):
        node = Node("a", 0, supplier="b", review_period=3, order_lead_time=1,
                    transit=Deterministic(4.0))
        self.assertAlmostEqual(node.expected_lead_time, 5.0)
        self.assertAlmostEqual(node.protection_interval, 7.0)

    def test_cumulative_protection_increases_upstream(self):
        network = build_chain()
        retailer = network.cumulative_protection("retailer")
        distributor = network.cumulative_protection("distributor")
        factory = network.cumulative_protection("factory")
        self.assertLess(retailer, distributor)
        self.assertLess(distributor, factory)
        self.assertAlmostEqual(distributor, retailer + network.nodes["distributor"].protection_interval)


class TestEchelonAccounting(unittest.TestCase):
    """The bug that cost the most to find: counting a downstream node's
    not-yet-shipped order as echelon inventory."""

    def setUp(self):
        self.network = build_chain()
        self.network.reset(100.0)
        for node in self.network.nodes.values():
            node.on_hand = 0.0

    def test_echelon_position_sums_physical_stock_downstream(self):
        self.network.nodes["retailer"].on_hand = 40.0
        self.network.nodes["distributor"].on_hand = 70.0
        self.network.nodes["factory"].on_hand = 90.0
        self.assertAlmostEqual(self.network.echelon_inventory_position("factory"), 200.0)
        self.assertAlmostEqual(self.network.echelon_inventory_position("distributor"), 110.0)
        self.assertAlmostEqual(self.network.echelon_inventory_position("retailer"), 40.0)

    def test_downstream_unshipped_orders_are_not_echelon_inventory(self):
        self.network.nodes["distributor"].on_hand = 100.0
        self.network.nodes["retailer"].on_order = 60.0  # ordered, still on the distributor's shelf
        self.assertAlmostEqual(self.network.echelon_inventory_position("distributor"), 100.0)

    def test_in_transit_units_between_echelon_members_count_once(self):
        self.network.nodes["distributor"].on_hand = 40.0
        self.network.nodes["retailer"].in_transit = 60.0  # dispatched, on the road
        self.assertAlmostEqual(self.network.echelon_inventory_position("distributor"), 100.0)

    def test_this_nodes_own_order_book_counts(self):
        self.network.nodes["factory"].on_order = 250.0
        self.assertAlmostEqual(self.network.echelon_inventory_position("factory"), 250.0)

    def test_end_customer_backorders_are_subtracted(self):
        self.network.nodes["retailer"].on_hand = 50.0
        self.network.nodes["retailer"].receive_order(None, 30.0, 0)
        self.assertAlmostEqual(self.network.echelon_inventory_position("distributor"), 20.0)

    def test_internal_backorders_are_not_subtracted(self):
        self.network.nodes["distributor"].on_hand = 50.0
        self.network.nodes["distributor"].receive_order("retailer", 30.0, 0)
        self.assertAlmostEqual(self.network.echelon_inventory_position("distributor"), 50.0)

    def test_installation_position_is_on_hand_less_backlog_plus_pipeline(self):
        node = self.network.nodes["retailer"]
        node.on_hand = 80.0
        node.on_order = 30.0
        node.in_transit = 20.0
        node.receive_order(None, 25.0, 0)
        self.assertAlmostEqual(node.inventory_position(), 80.0 - 25.0 + 50.0)
        self.assertAlmostEqual(node.outstanding, 50.0)


class TestConservation(unittest.TestCase):
    def _run(self, **kwargs):
        network = build_chain(**kwargs)
        simulator = Simulator(network, IIDNormal(100.0, 20.0), periods=300,
                              streams=StreamBank(seed=99))
        return simulator, simulator.run()

    def test_units_are_neither_created_nor_destroyed(self):
        """For each node: opening stock + everything received - everything
        shipped = closing stock. In a serial chain 'everything received' is the
        supplier's shipments less what is still on the road."""
        simulator, result = self._run()
        for node in simulator.network.stocking_nodes():
            supplier = simulator.network.nodes[node.supplier]
            received = result.nodes[supplier.name].shipped.sum() - node.in_transit
            opening = node.initial_periods_of_stock * 100.0
            shipped = result.nodes[node.name].shipped.sum()
            self.assertAlmostEqual(opening + received - shipped, node.on_hand, places=6)

    def test_backlog_equals_cumulative_demand_less_cumulative_shipments(self):
        simulator, result = self._run()
        for node in simulator.network.stocking_nodes():
            series = result.nodes[node.name]
            expected = series.demand_received.sum() - series.shipped.sum()
            self.assertAlmostEqual(expected, node.backlog_units, places=6)

    def test_on_hand_never_goes_negative(self):
        _simulator, result = self._run()
        for series in result.stocking_series():
            self.assertGreaterEqual(series.on_hand.min(), 0.0)

    def test_stochastic_lead_times_preserve_conservation(self):
        simulator, result = self._run(
            transit_factory=lambda level: GammaLeadTime(mean_lt=3.0, cv_lt=0.6, minimum=0.5)
        )
        for node in simulator.network.stocking_nodes():
            series = result.nodes[node.name]
            self.assertAlmostEqual(
                series.demand_received.sum() - series.shipped.sum(),
                node.backlog_units,
                places=6,
            )


class TestTimingConvention(unittest.TestCase):
    def test_oracle_base_stock_reproduces_demand_exactly(self):
        """With a known mean, no safety term, no batching and weekly review of
        one period, the order-up-to policy must place an order equal to the
        demand it just served. Any deviation is an off-by-one in the phase
        ordering -- this is the sharpest test of the period structure in the
        suite."""
        network = serial_chain(
            levels=1,
            policy_factory=lambda level: BaseStock(z=0.0, allow_returns=True),
            forecaster_factory=lambda level: Oracle(mean=100.0, std=20.0),
            transit_factory=lambda level: Deterministic(2.0),
        )
        result = run_simulation(network, IIDNormal(100.0, 20.0), periods=400, seed=5)
        series = result.nodes["retailer"]
        # Skip the opening transient during which the policy is drawing the
        # initial overstock down.
        np.testing.assert_allclose(
            series.orders_placed[60:], series.demand_received[60:], atol=1e-8
        )

    def test_a_receipt_arriving_in_a_period_is_shippable_that_period(self):
        network = serial_chain(levels=1, transit_factory=lambda level: Deterministic(2.0),
                               initial_periods_of_stock=0.0)
        result = run_simulation(network, IIDNormal(100.0, 0.0), periods=20, seed=1)
        series = result.nodes["retailer"]
        # Opening stock is zero, order lead 1 + transit 2, so the first receipt
        # lands in period 3 and must be shipped in period 3.
        self.assertEqual(series.shipped[:3].sum(), 0.0)
        self.assertGreater(series.shipped[3], 0.0)


class TestDeterminismAndCRN(unittest.TestCase):
    def test_same_seed_reproduces_the_run_exactly(self):
        first = run_simulation(build_chain(), IIDNormal(), periods=250, seed=3)
        second = run_simulation(build_chain(), IIDNormal(), periods=250, seed=3)
        for name in first.nodes:
            np.testing.assert_array_equal(
                first.nodes[name].orders_placed, second.nodes[name].orders_placed
            )

    def test_structural_change_does_not_move_the_demand_path(self):
        """Common random numbers: swapping the forecaster must leave every
        customer demand draw untouched."""
        moving = run_simulation(
            build_chain(forecaster_factory=lambda level: MovingAverage(10)),
            IIDNormal(), periods=250, seed=8,
        )
        smoothing = run_simulation(
            build_chain(forecaster_factory=lambda level: ExponentialSmoothing(0.4)),
            IIDNormal(), periods=250, seed=8,
        )
        np.testing.assert_array_equal(moving.customer_demand, smoothing.customer_demand)
        self.assertGreater(
            abs(moving.nodes["factory"].orders_placed - smoothing.nodes["factory"].orders_placed).sum(),
            0.0,
        )


class TestAllocationAndCapacity(unittest.TestCase):
    def test_proportional_allocation_shares_shortage_evenly(self):
        network = divergent_network(n_retailers=3, initial_periods_of_stock=0.5)
        network.nodes["distributor"].allocation = Allocation.PROPORTIONAL
        result = run_simulation(network, IIDNormal(100.0, 20.0), periods=200, seed=4)
        fills = [result.nodes[f"retailer{i}"].fill_rate() for i in (1, 2, 3)]
        self.assertLess(max(fills) - min(fills), 0.08)

    def test_capacity_caps_throughput(self):
        network = build_chain(capacity=60.0)
        result = run_simulation(network, IIDNormal(100.0, 10.0), periods=200, seed=6)
        self.assertLessEqual(result.nodes["factory"].shipped.max(), 60.0 + 1e-9)

    def test_capacity_loss_reduces_shipments_in_its_window(self):
        network = build_chain(capacity=150.0)
        plan = DisruptionPlan(
            capacity_losses=(CapacityLoss(node="factory", start=100, duration=20, factor=0.4),)
        )
        result = run_simulation(network, IIDNormal(100.0, 10.0), periods=200, seed=6,
                                disruptions=plan)
        self.assertLessEqual(result.nodes["factory"].shipped[100:120].max(), 60.0 + 1e-9)
        self.assertGreater(result.nodes["factory"].shipped[130:160].max(), 60.0)


class TestDisruptionMechanics(unittest.TestCase):
    def test_supply_outage_stops_shipments_for_exactly_its_duration(self):
        network = build_chain()
        plan = DisruptionPlan(outages=(SupplyOutage(node="source", start=80, duration=6),))
        result = run_simulation(network, IIDNormal(100.0, 10.0), periods=200, seed=2,
                                disruptions=plan)
        source = result.nodes["source"]
        self.assertEqual(source.shipped[80:86].sum(), 0.0)
        self.assertGreater(source.shipped[86:96].sum(), 0.0)

    def test_outage_backlog_is_worked_off_not_lost(self):
        network = build_chain()
        plan = DisruptionPlan(outages=(SupplyOutage(node="source", start=80, duration=6),))
        result = run_simulation(network, IIDNormal(100.0, 10.0), periods=300, seed=2,
                                disruptions=plan)
        factory = result.nodes["factory"]
        undisrupted = run_simulation(build_chain(), IIDNormal(100.0, 10.0), periods=300, seed=2)
        # Same demand, so the same total units must eventually flow through.
        self.assertAlmostEqual(
            factory.demand_received.sum(),
            undisrupted.nodes["factory"].demand_received.sum(),
            delta=0.15 * undisrupted.nodes["factory"].demand_received.sum(),
        )

    def test_overlapping_outages_on_one_node_are_rejected(self):
        plan = DisruptionPlan(
            outages=(
                SupplyOutage(node="source", start=50, duration=10),
                SupplyOutage(node="source", start=55, duration=5),
            )
        )
        with self.assertRaises(ValueError):
            Simulator(build_chain(), IIDNormal(), periods=100, disruptions=plan)

    def test_capacity_loss_on_an_uncapacitated_node_is_rejected(self):
        plan = DisruptionPlan(
            capacity_losses=(CapacityLoss(node="factory", start=10, duration=5),)
        )
        with self.assertRaises(ValueError):
            Simulator(build_chain(), IIDNormal(), periods=100, disruptions=plan)


class TestInformationModes(unittest.TestCase):
    def test_vmi_uses_the_echelon_position(self):
        network = build_chain(info_mode=InfoMode.VMI)
        network.reset(100.0)
        network.nodes["retailer"].on_hand = 50.0
        network.nodes["distributor"].on_hand = 50.0
        self.assertAlmostEqual(network.position_for_policy("distributor"), 100.0)

    def test_decentralised_uses_the_installation_position(self):
        network = build_chain(info_mode=InfoMode.DECENTRALIZED)
        network.reset(100.0)
        network.nodes["retailer"].on_hand = 50.0
        network.nodes["distributor"].on_hand = 50.0
        self.assertAlmostEqual(network.position_for_policy("distributor"), 50.0)

    def test_pos_sharing_flattens_upstream_amplification(self):
        common = dict(forecaster_factory=lambda level: ExponentialSmoothing(0.3))
        decentralised = run_simulation(
            build_chain(info_mode=InfoMode.DECENTRALIZED, **common),
            IIDNormal(), periods=600, seed=13,
        ).trim(100)
        shared = run_simulation(
            build_chain(info_mode=InfoMode.POS_SHARED, **common),
            IIDNormal(), periods=600, seed=13,
        ).trim(100)
        self.assertLess(
            shared.cumulative_bullwhip()["factory"],
            0.5 * decentralised.cumulative_bullwhip()["factory"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
