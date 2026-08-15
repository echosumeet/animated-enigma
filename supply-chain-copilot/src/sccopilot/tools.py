"""Tool registry with pydantic-typed argument schemas.

Five tools is enough to exercise every interesting agent behaviour: one open
text-to-SQL escape hatch and four narrow, pre-validated analytics tools that a
planner would actually want. Argument validation happens before any SQL runs,
so a malformed model call fails as a tool error the agent can recover from
rather than as an exception.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from .guard import SQLGuardError, run_guarded_sql

DEMAND_WINDOW_DAYS = 60
RISK_HORIZON_DAYS = 14


class ToolError(Exception):
    """Recoverable failure. The agent sees the message and may try again."""


class SQLQueryArgs(BaseModel):
    sql: str = Field(description="A single read-only SELECT/WITH statement.")
    limit: int = Field(default=100, ge=1, le=200)


class InventoryPositionArgs(BaseModel):
    sku_id: str = Field(description="e.g. SKU-0007")
    location_id: str | None = Field(default=None, description="e.g. DC-EAST; omit for network total")


class StockoutRiskArgs(BaseModel):
    location_id: str | None = None
    top_n: int = Field(default=5, ge=1, le=25)


class SupplierLeadTimeArgs(BaseModel):
    supplier_id: str | None = Field(default=None, description="e.g. SUP-03; omit for all suppliers")


class WhatIfArgs(BaseModel):
    sku_id: str
    location_id: str | None = None
    demand_uplift_pct: float = Field(default=0.0, ge=-90.0, le=500.0)
    lead_time_delta_days: float = Field(default=0.0, ge=-60.0, le=180.0)


def _latest_snapshot_day(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT MAX(date_id) FROM fact_inventory_snapshot").fetchone()[0]


def _avg_daily_demand(conn: sqlite3.Connection, sku_id: str, location_id: str | None) -> float:
    max_day = conn.execute("SELECT MAX(date_id) FROM fact_orders").fetchone()[0]
    lo = max(0, max_day - DEMAND_WINDOW_DAYS + 1)
    sql = "SELECT COALESCE(SUM(qty_ordered),0) FROM fact_orders WHERE sku_id=? AND date_id>=?"
    params: list[Any] = [sku_id, lo]
    if location_id:
        sql += " AND location_id=?"
        params.append(location_id)
    total = conn.execute(sql, params).fetchone()[0]
    return round(total / float(max_day - lo + 1), 4)


def tool_sql_query(conn: sqlite3.Connection, args: SQLQueryArgs) -> dict:
    try:
        res = run_guarded_sql(conn, args.sql, limit=args.limit)
    except SQLGuardError as exc:
        raise ToolError(str(exc)) from exc
    return {"sql": res.sql, "columns": res.columns, "rows": res.rows,
            "row_count": len(res.rows), "truncated": res.truncated, "notes": res.notes}


def tool_inventory_position(conn: sqlite3.Connection, args: InventoryPositionArgs) -> dict:
    day = _latest_snapshot_day(conn)
    sql = ("SELECT COALESCE(SUM(on_hand),0), COALESCE(SUM(on_order),0), COALESCE(SUM(safety_stock),0) "
           "FROM fact_inventory_snapshot WHERE date_id=? AND sku_id=?")
    params: list[Any] = [day, args.sku_id]
    if args.location_id:
        sql += " AND location_id=?"
        params.append(args.location_id)
    on_hand, on_order, ss = conn.execute(sql, params).fetchone()
    if on_hand == 0 and on_order == 0 and not conn.execute(
            "SELECT 1 FROM dim_sku WHERE sku_id=?", [args.sku_id]).fetchone():
        raise ToolError(f"unknown sku_id '{args.sku_id}'")
    demand = _avg_daily_demand(conn, args.sku_id, args.location_id)
    cover = round(on_hand / demand, 2) if demand > 0 else None
    return {"sku_id": args.sku_id, "location_id": args.location_id or "ALL", "as_of_day": day,
            "on_hand": on_hand, "on_order": on_order, "safety_stock": ss,
            "avg_daily_demand": demand, "days_of_cover": cover}


def tool_stockout_risk(conn: sqlite3.Connection, args: StockoutRiskArgs) -> dict:
    day = _latest_snapshot_day(conn)
    sql = ("SELECT sku_id, SUM(on_hand) AS oh, SUM(on_order) AS oo "
           "FROM fact_inventory_snapshot WHERE date_id=?")
    params: list[Any] = [day]
    if args.location_id:
        sql += " AND location_id=?"
        params.append(args.location_id)
    rows = conn.execute(sql + " GROUP BY sku_id", params).fetchall()
    out = []
    for sku_id, oh, oo in rows:
        demand = _avg_daily_demand(conn, sku_id, args.location_id)
        need = demand * RISK_HORIZON_DAYS
        # No observed demand means no stockout exposure, whatever the on-hand is.
        risk = max(0.0, min(1.0, 1.0 - (oh + oo) / need)) if need > 0 else 0.0
        out.append({"sku_id": sku_id, "on_hand": oh, "on_order": oo,
                    "expected_demand_14d": round(need, 1), "risk_score": round(risk, 3)})
    out.sort(key=lambda r: (-r["risk_score"], r["sku_id"]))
    return {"location_id": args.location_id or "ALL", "horizon_days": RISK_HORIZON_DAYS,
            "as_of_day": day, "at_risk": out[: args.top_n]}


def tool_supplier_lead_time_stats(conn: sqlite3.Connection, args: SupplierLeadTimeArgs) -> dict:
    sql = ("SELECT supplier_id, COUNT(*), AVG(promised_lead_days), AVG(actual_lead_days), "
           "AVG(CASE WHEN actual_lead_days<=promised_lead_days THEN 1.0 ELSE 0.0 END) "
           "FROM fact_shipments")
    params: list[Any] = []
    if args.supplier_id:
        sql += " WHERE supplier_id=?"
        params.append(args.supplier_id)
    rows = conn.execute(sql + " GROUP BY supplier_id ORDER BY supplier_id", params).fetchall()
    if not rows:
        raise ToolError(f"no shipments for supplier '{args.supplier_id}'")
    stats = [{"supplier_id": s, "shipments": n, "avg_promised_lead_days": round(p, 2),
              "avg_actual_lead_days": round(a, 2), "on_time_rate": round(o, 4),
              "lead_time_slip_days": round(a - p, 2)} for s, n, p, a, o in rows]
    worst = min(stats, key=lambda r: (r["on_time_rate"], r["supplier_id"]))
    return {"suppliers": stats, "worst_on_time_supplier": worst["supplier_id"],
            "worst_on_time_rate": worst["on_time_rate"]}


def tool_run_what_if(conn: sqlite3.Connection, args: WhatIfArgs) -> dict:
    pos = tool_inventory_position(
        conn, InventoryPositionArgs(sku_id=args.sku_id, location_id=args.location_id))
    base_demand = pos["avg_daily_demand"]
    new_demand = round(base_demand * (1.0 + args.demand_uplift_pct / 100.0), 4)
    base_cover = pos["days_of_cover"]
    new_cover = round(pos["on_hand"] / new_demand, 2) if new_demand > 0 else None
    # Exposure is measured against the replenishment lead time, not against a
    # fixed horizon: the question is whether cover outlasts the next receipt.
    base_lt = conn.execute(
        "SELECT AVG(actual_lead_days) FROM fact_shipments WHERE sku_id=?", [args.sku_id]).fetchone()[0]
    base_lt = float(base_lt) if base_lt is not None else 14.0
    new_lt = max(0.0, base_lt + args.lead_time_delta_days)
    shortfall = round(max(0.0, new_demand * new_lt - pos["on_hand"] - pos["on_order"]), 1)
    return {"sku_id": args.sku_id, "location_id": pos["location_id"],
            "base_daily_demand": base_demand, "scenario_daily_demand": new_demand,
            "base_days_of_cover": base_cover, "scenario_days_of_cover": new_cover,
            "base_lead_time_days": round(base_lt, 2), "scenario_lead_time_days": round(new_lt, 2),
            "projected_shortfall_units": shortfall,
            "stockout_expected": bool(shortfall > 0)}


class Tool:
    def __init__(self, name: str, description: str, schema: type[BaseModel],
                 fn: Callable[[sqlite3.Connection, Any], dict]):
        self.name, self.description, self.schema, self.fn = name, description, schema, fn

    def json_schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.schema.model_json_schema()}

    def call(self, conn: sqlite3.Connection, raw_args: dict) -> dict:
        try:
            parsed = self.schema.model_validate(raw_args or {})
        except ValidationError as exc:
            raise ToolError(f"invalid arguments for {self.name}: "
                            f"{'; '.join(e['msg'] for e in exc.errors())}") from exc
        return self.fn(conn, parsed)


class ToolRegistry:
    def __init__(self, conn: sqlite3.Connection, tools: list[Tool]):
        self.conn = conn
        self.tools = {t.name: t for t in tools}

    def specs(self) -> list[dict]:
        return [t.json_schema() for t in self.tools.values()]

    def call(self, name: str, args: dict) -> dict:
        if name not in self.tools:
            raise ToolError(f"unknown tool '{name}'; available: {', '.join(sorted(self.tools))}")
        return self.tools[name].call(self.conn, args)


def default_registry(conn: sqlite3.Connection) -> ToolRegistry:
    return ToolRegistry(conn, [
        Tool("sql_query", "Run one read-only SELECT against the planning warehouse.",
             SQLQueryArgs, tool_sql_query),
        Tool("inventory_position", "On hand, on order, safety stock and days of cover for a SKU.",
             InventoryPositionArgs, tool_inventory_position),
        Tool("stockout_risk", "Rank SKUs by 14-day stockout risk at a location.",
             StockoutRiskArgs, tool_stockout_risk),
        Tool("supplier_lead_time_stats", "Promised vs actual lead time and on-time rate by supplier.",
             SupplierLeadTimeArgs, tool_supplier_lead_time_stats),
        Tool("run_what_if", "Re-project cover and shortfall under a demand or lead-time shock.",
             WhatIfArgs, tool_run_what_if),
    ])
