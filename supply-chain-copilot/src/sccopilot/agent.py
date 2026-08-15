"""The agent loop: plan -> tool call -> observe -> respond.

Deliberately small and boring. The interesting parts of an agent in an
operational setting are the step budget, what happens when a tool errors, and
whether you can reconstruct afterwards why it said what it said - so those are
the parts that are explicit here.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from .guard import schema_prompt
from .providers import LLMProvider, Message, StubProvider
from .tools import ToolError, ToolRegistry, default_registry

SYSTEM_PROMPT = f"""You are a supply chain planning copilot with read-only access to a
planning warehouse. Use the tools; never state a number you did not get from a tool.
If the question does not identify the SKU, location or supplier, ask one clarifying
question instead of guessing. If the question is outside the warehouse, say so.

{schema_prompt()}
"""

MAX_STEPS = 6


@dataclass
class AgentResult:
    question: str
    answer: str
    tools_used: list[str] = field(default_factory=list)
    steps: int = 0
    tool_errors: int = 0
    latency_ms: float = 0.0
    trace: list[dict] = field(default_factory=list)
    stopped_on_budget: bool = False

    def trace_json(self, indent: int = 2) -> str:
        return json.dumps(self.trace, indent=indent, default=str)


class Agent:
    def __init__(self, conn: sqlite3.Connection, provider: LLMProvider | None = None,
                 registry: ToolRegistry | None = None, max_steps: int = MAX_STEPS):
        self.conn = conn
        self.provider = provider or StubProvider()
        self.registry = registry or default_registry(conn)
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        messages = [Message("system", SYSTEM_PROMPT), Message("user", question)]
        specs = self.registry.specs()
        result = AgentResult(question=question, answer="")
        t_run = time.perf_counter()

        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            completion = self.provider.complete(messages, specs)
            plan_ms = round((time.perf_counter() - t0) * 1000, 3)

            if not completion.tool_calls:
                result.answer = completion.text or ""
                result.steps = step
                result.trace.append({"step": step, "kind": "final", "latency_ms": plan_ms,
                                     "answer": result.answer})
                break

            call = completion.tool_calls[0]
            result.trace.append({"step": step, "kind": "tool_call", "tool": call.name,
                                 "arguments": call.arguments, "latency_ms": plan_ms})
            result.tools_used.append(call.name)
            t1 = time.perf_counter()
            try:
                observation = self.registry.call(call.name, call.arguments)
            except ToolError as exc:
                tool_ms = round((time.perf_counter() - t1) * 1000, 3)
                result.tool_errors += 1
                messages.append(Message("tool", tool_name=call.name, tool_error=str(exc)))
                result.trace.append({"step": step, "kind": "observation", "tool": call.name,
                                     "ok": False, "error": str(exc), "latency_ms": tool_ms})
                continue
            tool_ms = round((time.perf_counter() - t1) * 1000, 3)
            messages.append(Message("tool", tool_name=call.name, tool_result=observation))
            result.trace.append({"step": step, "kind": "observation", "tool": call.name,
                                 "ok": True, "latency_ms": tool_ms,
                                 "result": _summarise(observation)})
            result.steps = step
        else:
            result.stopped_on_budget = True
            result.answer = (f"I stopped after the {self.max_steps}-step budget without a "
                             f"grounded answer. Narrow the question and ask again.")
            result.trace.append({"step": self.max_steps, "kind": "budget_exhausted",
                                 "latency_ms": 0.0})

        result.latency_ms = round((time.perf_counter() - t_run) * 1000, 3)
        return result


def _summarise(obs: dict, max_chars: int = 320) -> str:
    """Traces keep a truncated observation, not the full payload - a real trace
    store cannot afford to hold every row a tool returned."""
    text = json.dumps(obs, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + "...<truncated>"
