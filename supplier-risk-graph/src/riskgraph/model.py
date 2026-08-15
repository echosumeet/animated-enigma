"""Typed multi-tier supply network model.

Nodes are parts, sites and suppliers. A part is consumed by a parent part through a
BOM edge with a quantity-per; a part is produced at one or more sites through a supply
edge carrying an allocation share. Sites belong to suppliers and carry the geography
and the disruption parameters, because disruption is a property of a physical location,
not of a legal entity.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
import networkx as nx
from pydantic import BaseModel, Field, model_validator

class Part(BaseModel):
    part_id: str
    name: str
    tier: int = Field(ge=0)
    unit_cost: float = Field(gt=0)
    annual_units: float = Field(default=0.0, ge=0)
    annual_revenue: float = Field(default=0.0, ge=0)
    finished: bool = False


class Supplier(BaseModel):
    supplier_id: str
    name: str
    tier: int = Field(ge=1)
    financial_health: float = Field(default=0.7, ge=0, le=1)


class Site(BaseModel):
    site_id: str
    supplier_id: str
    country: str
    region: str
    weekly_capacity: float = Field(gt=0)
    disruption_rate: float = Field(ge=0, le=1)
    mean_recovery_days: float = Field(gt=0)


class BomEdge(BaseModel):
    parent: str
    child: str
    qty_per: float = Field(gt=0)


class SupplyEdge(BaseModel):
    part_id: str
    site_id: str
    share: float = Field(gt=0, le=1)
    lead_time_days: float = Field(default=30.0, gt=0)


class SupplyNetwork(BaseModel):
    """A validated multi-tier network. Validation is where most real loaders fail."""

    parts: list[Part]
    suppliers: list[Supplier]
    sites: list[Site]
    bom: list[BomEdge]
    supply: list[SupplyEdge]

    @model_validator(mode="after")
    def _validate(self) -> "SupplyNetwork":
        part_ids = {p.part_id for p in self.parts}
        sup_ids = {s.supplier_id for s in self.suppliers}
        site_ids = {s.site_id for s in self.sites}
        if len(part_ids) != len(self.parts) or len(site_ids) != len(self.sites):
            raise ValueError("duplicate part or site identifiers")
        for s in self.sites:
            if s.supplier_id not in sup_ids:
                raise ValueError(f"site {s.site_id} references unknown supplier {s.supplier_id}")
        for e in self.bom:
            if e.parent not in part_ids or e.child not in part_ids:
                raise ValueError(f"bom edge {e.parent}->{e.child} references unknown part")
            if e.parent == e.child:
                raise ValueError(f"self-referencing bom edge on {e.parent}")
        for e in self.supply:
            if e.part_id not in part_ids:
                raise ValueError(f"supply edge references unknown part {e.part_id}")
            if e.site_id not in site_ids:
                raise ValueError(f"supply edge references unknown site {e.site_id}")

        by_part: dict[str, float] = {}
        for e in self.supply:
            by_part[e.part_id] = by_part.get(e.part_id, 0.0) + e.share
        for pid, total in by_part.items():
            if abs(total - 1.0) > 1e-4:
                raise ValueError(f"allocation shares for {pid} sum to {total:.4f}, expected 1.0")

        g = nx.DiGraph((e.parent, e.child) for e in self.bom)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("BOM contains a cycle")
        if not any(p.finished for p in self.parts):
            raise ValueError("network has no finished good")
        return self

    # -- indexes ---------------------------------------------------------------
    def part(self, pid: str) -> Part:
        return self._parts[pid]

    def site(self, sid: str) -> Site:
        return self._sites[sid]

    def supplier(self, sid: str) -> Supplier:
        return self._suppliers[sid]

    @cached_property
    def _parts(self) -> dict[str, Part]:
        return {p.part_id: p for p in self.parts}

    @cached_property
    def _sites(self) -> dict[str, Site]:
        return {s.site_id: s for s in self.sites}

    @cached_property
    def _suppliers(self) -> dict[str, Supplier]:
        return {s.supplier_id: s for s in self.suppliers}

    @cached_property
    def _children(self) -> dict[str, list[BomEdge]]:
        out: dict[str, list[BomEdge]] = {p.part_id: [] for p in self.parts}
        for e in self.bom:
            out[e.parent].append(e)
        return out

    @cached_property
    def _sources(self) -> dict[str, list[SupplyEdge]]:
        out: dict[str, list[SupplyEdge]] = {p.part_id: [] for p in self.parts}
        for e in self.supply:
            out[e.part_id].append(e)
        return out

    def children(self, pid: str) -> list[BomEdge]:
        return self._children[pid]

    def sources(self, pid: str) -> list[SupplyEdge]:
        return self._sources[pid]

    def finished_goods(self) -> list[Part]:
        return [p for p in self.parts if p.finished]

    def site_supplier(self, site_id: str) -> str:
        return self._sites[site_id].supplier_id

    def supplier_sites(self, supplier_id: str) -> list[str]:
        return [s.site_id for s in self.sites if s.supplier_id == supplier_id]

    # -- graph -----------------------------------------------------------------
    def to_graph(self) -> nx.DiGraph:
        """Directed graph: finished good -> components -> sites -> suppliers."""
        g = nx.DiGraph()
        for p in self.parts:
            g.add_node(f"part:{p.part_id}", kind="part", tier=p.tier, label=p.part_id)
        for s in self.sites:
            g.add_node(f"site:{s.site_id}", kind="site", country=s.country, label=s.site_id)
        for s in self.suppliers:
            g.add_node(f"supplier:{s.supplier_id}", kind="supplier", tier=s.tier, label=s.supplier_id)
        for e in self.bom:
            g.add_edge(f"part:{e.parent}", f"part:{e.child}", kind="bom", qty_per=e.qty_per)
        for e in self.supply:
            g.add_edge(f"part:{e.part_id}", f"site:{e.site_id}", kind="supply", share=e.share)
        for s in self.sites:
            g.add_edge(f"site:{s.site_id}", f"supplier:{s.supplier_id}", kind="operated_by")
        return g

    # -- io --------------------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(), indent=2))


def load_network(path: str | Path) -> SupplyNetwork:
    """Load and validate a network from JSON. Raises on any structural defect."""
    return SupplyNetwork.model_validate(json.loads(Path(path).read_text()))
