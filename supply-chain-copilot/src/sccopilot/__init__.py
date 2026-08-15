"""sccopilot - a guarded, evaluated supply chain planning copilot."""

from .agent import Agent, AgentResult
from .evals import GOLDEN_SET, run_attack_suite, run_eval
from .guard import SQLGuardError, run_guarded_sql, schema_prompt
from .providers import AnthropicProvider, Completion, Message, OpenAIProvider, StubProvider
from .tools import ToolError, ToolRegistry, default_registry
from .warehouse import build_warehouse, row_counts

__version__ = "0.1.0"
__all__ = [
    "Agent", "AgentResult", "GOLDEN_SET", "run_eval", "run_attack_suite",
    "SQLGuardError", "run_guarded_sql", "schema_prompt", "StubProvider",
    "AnthropicProvider", "OpenAIProvider", "Message", "Completion",
    "ToolError", "ToolRegistry", "default_registry", "build_warehouse", "row_counts",
]
