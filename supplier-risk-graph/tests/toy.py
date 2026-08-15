"""A hand-built network with a known answer, used by several test modules.

FG -> SA1, SA2 (two tier-1 branches, each dual sourced)
SA1 -> C1 -> M1, MSHARED
SA2 -> C2 -> MSHARED
Everything is dual sourced except MSHARED, which is sole sourced at SITE-X. Both
tier-1 branches therefore hide the same diamond, and only SITE-X can stop the FG
outright.
"""

from riskgraph.model import BomEdge, Part, Site, Supplier, SupplyEdge, SupplyNetwork


def toy_network() -> SupplyNetwork:
    parts = [
        Part(part_id="FG", name="fg", tier=0, unit_cost=500.0, annual_units=1000.0,
             annual_revenue=1_000_000.0, finished=True),
        Part(part_id="SA1", name="sa1", tier=1, unit_cost=100.0),
        Part(part_id="SA2", name="sa2", tier=1, unit_cost=120.0),
        Part(part_id="C1", name="c1", tier=2, unit_cost=20.0),
        Part(part_id="C2", name="c2", tier=2, unit_cost=25.0),
        Part(part_id="M1", name="m1", tier=3, unit_cost=2.0),
        Part(part_id="MSHARED", name="shared", tier=3, unit_cost=3.0),
    ]
    bom = [
        BomEdge(parent="FG", child="SA1", qty_per=1.0),
        BomEdge(parent="FG", child="SA2", qty_per=2.0),
        BomEdge(parent="SA1", child="C1", qty_per=3.0),
        BomEdge(parent="SA2", child="C2", qty_per=1.0),
        BomEdge(parent="C1", child="M1", qty_per=2.0),
        BomEdge(parent="C1", child="MSHARED", qty_per=1.0),
        BomEdge(parent="C2", child="MSHARED", qty_per=4.0),
    ]
    suppliers = [
        Supplier(supplier_id="SUP-A", name="a", tier=1),
        Supplier(supplier_id="SUP-B", name="b", tier=2),
        Supplier(supplier_id="SUP-X", name="x", tier=3),
    ]
    sites = [
        Site(site_id="SITE-A1", supplier_id="SUP-A", country="DE", region="EMEA",
             weekly_capacity=1000.0, disruption_rate=0.05, mean_recovery_days=30.0),
        Site(site_id="SITE-A2", supplier_id="SUP-A", country="US", region="AMER",
             weekly_capacity=1000.0, disruption_rate=0.05, mean_recovery_days=30.0),
        Site(site_id="SITE-B1", supplier_id="SUP-B", country="CN", region="APAC",
             weekly_capacity=1000.0, disruption_rate=0.05, mean_recovery_days=40.0),
        Site(site_id="SITE-X", supplier_id="SUP-X", country="TW", region="APAC",
             weekly_capacity=1000.0, disruption_rate=0.10, mean_recovery_days=60.0),
    ]
    supply = [
        SupplyEdge(part_id="SA1", site_id="SITE-A1", share=0.6),
        SupplyEdge(part_id="SA1", site_id="SITE-B1", share=0.4),
        SupplyEdge(part_id="SA2", site_id="SITE-A2", share=0.6),
        SupplyEdge(part_id="SA2", site_id="SITE-B1", share=0.4),
        SupplyEdge(part_id="C1", site_id="SITE-B1", share=0.6),
        SupplyEdge(part_id="C1", site_id="SITE-A1", share=0.4),
        SupplyEdge(part_id="C2", site_id="SITE-B1", share=0.7),
        SupplyEdge(part_id="C2", site_id="SITE-A2", share=0.3),
        SupplyEdge(part_id="M1", site_id="SITE-B1", share=0.5),
        SupplyEdge(part_id="M1", site_id="SITE-A1", share=0.5),
        SupplyEdge(part_id="MSHARED", site_id="SITE-X", share=1.0),
    ]
    return SupplyNetwork(parts=parts, suppliers=suppliers, sites=sites, bom=bom, supply=supply)
