"""LLM provider protocol plus a deterministic stub.

The stub is the default so that the tests, the eval harness and the examples all
run with no API key and no network. It is a rule-based planner, not a model: it
routes a question to a tool plan by pattern match and renders the final answer
from the observations. That is enough to exercise the agent loop, the guardrails
and the graders, which is what this repository is about. Swapping in
``AnthropicProvider`` or ``OpenAIProvider`` changes nothing above this line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_name: str | None = None
    tool_result: dict | None = None
    tool_error: str | None = None


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Completion:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, messages: list[Message], tools: list[dict]) -> Completion:
        ...


SKU_RE = re.compile(r"SKU-\d{4}", re.IGNORECASE)
LOC_RE = re.compile(r"DC-[A-Z]+", re.IGNORECASE)
SUP_RE = re.compile(r"SUP-\d{2}", re.IGNORECASE)
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct)")
DAYS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*days?")

OUT_OF_SCOPE = re.compile(
    r"\b(weather|poem|joke|share price|stock price|home address|who should i hire|"
    r"salary|holiday|recipe)\b", re.IGNORECASE)
ATTACK = re.compile(
    r"\b(drop|delete|attach|pragma|insert into|update\s+\w+\s+set|internal_credentials|"
    r"sqlite_master|ignore (all |your )?(previous |prior )?instructions)\b", re.IGNORECASE)

AGGREGATION_SQL: list[tuple[str, str]] = [
    (r"category (\w+)",
     "SELECT SUM(o.qty_ordered) AS total_units FROM fact_orders o "
     "JOIN dim_sku s ON s.sku_id=o.sku_id WHERE s.category='{0}'"),
    (r"order lines|how many rows",
     "SELECT COUNT(*) AS order_lines FROM fact_orders"),
    (r"late|behind promise",
     "SELECT COUNT(*) AS late_shipments FROM fact_shipments "
     "WHERE actual_lead_days > promised_lead_days"),
    (r"unit cost",
     "SELECT ROUND(AVG(unit_cost),4) AS avg_unit_cost FROM dim_sku WHERE abc_class='A'"),
    (r"total units|ordered across|total ordered",
     "SELECT SUM(qty_ordered) AS total_units FROM fact_orders"),
]

CLARIFY = ("I need one more detail before I can answer: name the SKU, location or supplier "
           "you mean (for example SKU-0007 at DC-EAST). I will not guess which entity you meant.")
REFUSE_SCOPE = ("That is out of scope for this copilot. I only answer questions the planning "
                "warehouse can support: inventory positions, orders, shipments and supplier "
                "lead times.")
REFUSE_ATTACK = ("Refused. That request tries to run a statement outside the read-only "
                 "allowlist, so it was blocked by the SQL guard and I will not retry it. "
                 "I can answer read-only questions about the allowlisted planning tables.")


def _step(name: str, **args) -> dict:
    return {"tool": name, "arguments": args}


class StubProvider:
    """Deterministic, pattern-matched provider. Default everywhere."""

    name = "stub"

    def complete(self, messages: list[Message], tools: list[dict]) -> Completion:
        question = next((m.content for m in messages if m.role == "user"), "")
        obs = [m for m in messages if m.role == "tool"]
        good = [m for m in obs if m.tool_error is None]
        route = self.route(question)

        if obs and obs[-1].tool_error is not None:
            if route["mode"] == "attack":
                return Completion(text=REFUSE_ATTACK)
            failed = obs[-1]
            retried = sum(1 for m in obs if m.tool_error is not None)
            if retried == 1 and failed.tool_name == "inventory_position":
                # Degrade gracefully: drop the optional location filter and retry once.
                sku = SKU_RE.search(question)
                if sku:
                    return Completion(tool_calls=[ToolCall("inventory_position",
                                                           {"sku_id": sku.group(0).upper()})])
            return Completion(text=f"I could not complete that: {failed.tool_error}. "
                                   f"Nothing was answered from a partial result.")

        if route["mode"] in ("clarify", "refuse"):
            return Completion(text=route["text"])

        steps = route["steps"]
        if len(good) < len(steps):
            spec = steps[len(good)]
            if callable(spec):
                spec = spec([m.tool_result for m in good])
            return Completion(tool_calls=[ToolCall(spec["tool"], spec["arguments"])],
                              stop_reason="tool_use")
        return Completion(text=route["render"]([m.tool_result for m in good]))

    # ------------------------------------------------------------------ routing
    def route(self, question: str) -> dict:
        q = question.strip()
        low = q.lower()
        sku = (SKU_RE.search(q).group(0).upper() if SKU_RE.search(q) else None)
        loc = (LOC_RE.search(q).group(0).upper() if LOC_RE.search(q) else None)
        sup = (SUP_RE.search(q).group(0).upper() if SUP_RE.search(q) else None)

        if ATTACK.search(low):
            m = re.search(r"(select .*)$", q, re.IGNORECASE | re.DOTALL)
            sql = m.group(1).strip() if m else "SELECT * FROM internal_credentials"
            return {"mode": "attack", "steps": [_step("sql_query", sql=sql)],
                    "render": lambda o: REFUSE_ATTACK}
        if OUT_OF_SCOPE.search(low):
            return {"mode": "refuse", "text": REFUSE_SCOPE, "steps": []}

        if "what if" in low or "slips" in low or "uplift" in low:
            if not sku:
                return {"mode": "clarify", "text": CLARIFY, "steps": []}
            pct = float(PCT_RE.search(low).group(1)) if PCT_RE.search(low) else 0.0
            days = float(DAYS_RE.search(low).group(1)) if DAYS_RE.search(low) else 0.0
            args = {"sku_id": sku, "demand_uplift_pct": pct, "lead_time_delta_days": days}
            if loc:
                args["location_id"] = loc
            return {"mode": "tool", "steps": [_step("run_what_if", **args)],
                    "render": _render_what_if}

        if "worst on-time" in low or "worst on time" in low:
            if "distinct" in low or "how many skus" in low:
                dep = lambda o: _step("sql_query", sql=(  # noqa: E731
                    "SELECT COUNT(DISTINCT sku_id) AS distinct_skus FROM fact_shipments "
                    f"WHERE supplier_id='{o[0]['worst_on_time_supplier']}'"))
            else:
                dep = lambda o: _step("sql_query", sql=(  # noqa: E731
                    "SELECT SUM(qty) AS total_qty FROM fact_shipments "
                    f"WHERE supplier_id='{o[0]['worst_on_time_supplier']}'"))
            return {"mode": "tool",
                    "steps": [_step("supplier_lead_time_stats"), dep],
                    "render": lambda o: (
                        f"Worst on-time supplier is {o[0]['worst_on_time_supplier']} at "
                        f"{o[0]['worst_on_time_rate']:.3f} on-time rate.\n" + _render_sql(o[1:]))}

        if "highest" in low and "risk" in low:
            args = {"location_id": loc, "top_n": 1} if loc else {"top_n": 1}
            if "unit cost" in low:
                dep = lambda o: _step("sql_query", sql=(  # noqa: E731
                    "SELECT unit_cost FROM dim_sku "
                    f"WHERE sku_id='{o[0]['at_risk'][0]['sku_id']}'"))
            else:
                dep = lambda o: _step("sql_query", sql=(  # noqa: E731
                    "SELECT SUM(qty_ordered) AS units_60d FROM fact_orders "
                    f"WHERE sku_id='{o[0]['at_risk'][0]['sku_id']}' "
                    "AND date_id >= (SELECT MAX(date_id)-59 FROM fact_orders)"))
            return {"mode": "tool", "steps": [_step("stockout_risk", **args), dep],
                    "render": lambda o: (
                        f"Highest-risk SKU is {o[0]['at_risk'][0]['sku_id']} "
                        f"(risk {o[0]['at_risk'][0]['risk_score']}).\n" + _render_sql(o[1:]))}

        if sup or "which supplier" in low or "all suppliers" in low:
            args = {"supplier_id": sup} if sup else {}
            return {"mode": "tool", "steps": [_step("supplier_lead_time_stats", **args)],
                    "render": lambda o: _render_supplier(o, low)}
        if re.search(r"\b(supplier|lead time|on-time)\b", low):
            return {"mode": "clarify", "text": CLARIFY, "steps": []}

        if "risk" in low or "stock out" in low or "stockout" in low:
            args = {"location_id": loc} if loc else {}
            return {"mode": "tool", "steps": [_step("stockout_risk", **args)],
                    "render": lambda o: (
                        f"Top stockout risk at {o[0]['location_id']}: "
                        f"{o[0]['at_risk'][0]['sku_id']}\nAnswer: "
                        f"{o[0]['at_risk'][0]['risk_score']}")}

        for pattern, template in AGGREGATION_SQL:
            m = re.search(pattern, low)
            if m and re.search(r"how many|total|average|count|sum", low):
                sql = template.format(*m.groups()) if m.groups() else template
                return {"mode": "tool", "steps": [_step("sql_query", sql=sql)],
                        "render": _render_sql}

        if re.search(r"on hand|on-hand|cover|inventory|stock|position", low):
            if not sku:
                return {"mode": "clarify", "text": CLARIFY, "steps": []}
            args = {"sku_id": sku, "location_id": loc} if loc else {"sku_id": sku}
            return {"mode": "tool", "steps": [_step("inventory_position", **args)],
                    "render": lambda o: _render_inventory(o, low)}

        return {"mode": "clarify", "text": CLARIFY, "steps": []}


def _render_sql(obs: list[dict]) -> str:
    o = obs[-1]
    if not o["rows"]:
        return "The query returned no rows.\nAnswer: 0"
    head = dict(zip(o["columns"], o["rows"][0]))
    body = ", ".join(f"{k} = {v}" for k, v in head.items())
    first_num = next((v for v in head.values() if isinstance(v, (int, float))), None)
    return f"{body} (rows returned: {o['row_count']})\nAnswer: {first_num}"


def _render_inventory(obs: list[dict], low: str) -> str:
    o = obs[-1]
    val = o["days_of_cover"] if "cover" in low else o["on_hand"]
    return (f"{o['sku_id']} at {o['location_id']} as of day {o['as_of_day']}: on hand {o['on_hand']}, "
            f"on order {o['on_order']}, safety stock {o['safety_stock']}, "
            f"days of cover {o['days_of_cover']}.\nAnswer: {val}")


def _render_supplier(obs: list[dict], low: str) -> str:
    o = obs[-1]
    s = o["suppliers"][0] if len(o["suppliers"]) == 1 else \
        min(o["suppliers"], key=lambda r: r["on_time_rate"])
    val = s["avg_actual_lead_days"] if "lead time" in low else s["on_time_rate"]
    return (f"{s['supplier_id']}: {s['shipments']} shipments, promised "
            f"{s['avg_promised_lead_days']} days, actual {s['avg_actual_lead_days']} days, "
            f"on-time rate {s['on_time_rate']}.\nAnswer: {val}")


def _render_what_if(obs: list[dict]) -> str:
    o = obs[-1]
    verdict = "a stockout is expected" if o["stockout_expected"] else "cover holds"
    return (f"{o['sku_id']} at {o['location_id']}: daily demand {o['base_daily_demand']} -> "
            f"{o['scenario_daily_demand']}, cover {o['base_days_of_cover']} -> "
            f"{o['scenario_days_of_cover']} days against a "
            f"{o['scenario_lead_time_days']}-day lead time, so {verdict}.\n"
            f"Answer: {o['projected_shortfall_units']}")


class AnthropicProvider:
    """Thin adapter. The SDK is imported inside the method so the package works
    without it installed; nothing in the tests or examples touches this path."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 1024):
        self.model, self.max_tokens = model, max_tokens

    def complete(self, messages: list[Message], tools: list[dict]) -> Completion:
        import anthropic  # noqa: PLC0415  (deliberate lazy import)

        client = anthropic.Anthropic()
        system = "\n".join(m.content for m in messages if m.role == "system")
        payload: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                payload.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": m.tool_name or "tool",
                     "content": str(m.tool_error or m.tool_result),
                     "is_error": m.tool_error is not None}]})
            else:
                payload.append({"role": m.role, "content": m.content})
        resp = client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                      system=system, tools=tools, messages=payload)
        calls = [ToolCall(b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        text = "".join(b.text for b in resp.content if b.type == "text") or None
        return Completion(text=text, tool_calls=calls, stop_reason=resp.stop_reason or "end_turn")


class OpenAIProvider:
    """Same contract against the chat-completions tool-calling shape."""

    name = "openai"

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def complete(self, messages: list[Message], tools: list[dict]) -> Completion:
        import json  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI()
        payload = []
        for m in messages:
            if m.role == "tool":
                payload.append({"role": "tool", "tool_call_id": m.tool_name or "tool",
                                "content": str(m.tool_error or m.tool_result)})
            else:
                payload.append({"role": m.role, "content": m.content})
        spec = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]
        resp = client.chat.completions.create(model=self.model, messages=payload, tools=spec)
        choice = resp.choices[0].message
        calls = [ToolCall(c.function.name, json.loads(c.function.arguments or "{}"))
                 for c in (choice.tool_calls or [])]
        return Completion(text=choice.content, tool_calls=calls,
                          stop_reason=resp.choices[0].finish_reason or "end_turn")
