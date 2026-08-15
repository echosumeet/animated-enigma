"""End-to-end run: generate, extract, reconcile, sweep. One entry point for the
benchmarks, the CLI, the figure and the example so they cannot drift apart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .generate import Corpus, NoiseConfig, build_corpus
from .llm import LLMProvider, StubProvider
from .pipeline import DocResult, document_correct, extract_document, field_correctness
from .reconcile import Discrepancy, evaluate_detection, reconcile
from .sweep import SweepPoint, field_accuracy_table, operating_point, sweep_thresholds


@dataclass
class RunOutput:
    corpus: Corpus
    results: list[DocResult]
    confidences: list[float]
    correct: list[bool]
    accuracy: dict[str, Any]
    points: list[SweepPoint]
    operating: Optional[SweepPoint]
    detection: dict[str, Any]
    discrepancies: dict[str, list[Discrepancy]]
    llm_calls: int


def run(n_shipments: int = 120, seed: int = 7, noise: Optional[NoiseConfig] = None,
        provider: Optional[LLMProvider] = None, error_budget: float = 0.05,
        confirm: str = "high_value") -> RunOutput:
    corpus = build_corpus(n_shipments=n_shipments, seed=seed, noise=noise)
    provider = provider or StubProvider()

    results: list[DocResult] = []
    truths: dict[tuple[str, str], dict] = {}
    per_doc_flags: list[tuple[str, dict[str, bool]]] = []
    confidences: list[float] = []
    correct: list[bool] = []
    llm_calls = 0

    for doc in corpus.docs:
        res = extract_document(doc, provider, confirm=confirm)
        results.append(res)
        truths[(doc.shipment_id, doc.doc_type)] = doc.truth
        per_doc_flags.append((doc.doc_type, field_correctness(res, doc.truth)))
        confidences.append(res.min_confidence())
        correct.append(document_correct(res, doc.truth))
        llm_calls += res.llm_calls

    by_shipment: dict[str, dict[str, DocResult]] = {}
    for res in results:
        by_shipment.setdefault(res.shipment_id, {})[res.doc_type] = res
    discrepancies = {sid: reconcile(docs) for sid, docs in by_shipment.items()}

    points = sweep_thresholds(confidences, correct)
    return RunOutput(
        corpus=corpus, results=results, confidences=confidences, correct=correct,
        accuracy=field_accuracy_table(per_doc_flags), points=points,
        operating=operating_point(points, error_budget),
        detection=evaluate_detection(corpus.seeded, discrepancies),
        discrepancies=discrepancies, llm_calls=llm_calls,
    )
