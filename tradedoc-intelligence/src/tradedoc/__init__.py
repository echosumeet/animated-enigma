"""tradedoc -- verifiable trade-document extraction.

The repository generates its own documents and their ground truth, so every accuracy,
confidence and reconciliation number can be recomputed from source.
"""

from .evaluate import RunOutput, run
from .generate import Corpus, GeneratedDoc, NoiseConfig, Shipment, build_corpus
from .llm import AnthropicProvider, LLMProvider, OpenAIProvider, StubProvider
from .pipeline import DocResult, FieldResult, document_correct, extract_document
from .reconcile import Discrepancy, evaluate_detection, reconcile
from .schemas import LineItem, TradeDocument, ValidationIssue
from .sweep import SweepPoint, operating_point, sweep_thresholds

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider", "Corpus", "DocResult", "Discrepancy", "FieldResult",
    "GeneratedDoc", "LLMProvider", "LineItem", "NoiseConfig", "OpenAIProvider",
    "RunOutput", "Shipment", "StubProvider", "SweepPoint", "TradeDocument",
    "ValidationIssue", "build_corpus", "document_correct", "evaluate_detection",
    "extract_document", "operating_point", "reconcile", "run", "sweep_thresholds",
    "__version__",
]
