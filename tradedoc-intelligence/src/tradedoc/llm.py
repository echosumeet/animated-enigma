"""LLM pass for fields the rules could not resolve.

The provider is a protocol with three implementations. ``StubProvider`` is the default
and is what every test and benchmark in this repo runs against: it is deterministic,
offline, and deliberately *different in kind* from the rule extractor -- a loose
keyword-proximity reader with no validation. That difference is the whole point. If the
second pass were a copy of the first, agreement between them would carry no information
and the confidence score built on it in ``pipeline.py`` would be a constant.

The hosted providers lazily import their SDKs inside the call, so importing this module
never requires a key or a network.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Protocol, runtime_checkable

FIELD_KEYWORDS: dict[str, list[str]] = {
    "doc_number": ["invoice no", "document number", "inv#", "packing list no", "b/l no"],
    "doc_date": ["date", "issued", "dated"],
    "buyer": ["sold to", "consignee", "bill to"],
    "currency": ["currency", "total"],
    "incoterm": ["terms of delivery", "incoterm"],
    "origin_country": ["country of origin"],
    "total_value": ["total"],
    "package_count": ["packages"],
    "net_weight_kg": ["net weight"],
    "gross_weight_kg": ["gross weight"],
    "vessel": ["vessel"],
    "port_of_loading": ["port of loading"],
    "port_of_discharge": ["port of discharge"],
    "container_no": ["container"],
}

_NUMERIC_FIELDS = {"total_value", "package_count", "net_weight_kg", "gross_weight_kg"}
_CODE_FIELDS = {"currency", "incoterm", "origin_country"}


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal surface: one residual field at a time, string in, string or None out."""

    name: str

    def extract_field(self, text: str, field: str, doc_type: str) -> Optional[str]:
        ...


class StubProvider:
    """Offline, deterministic keyword-proximity reader.

    ``miss_rate`` is applied through a content hash rather than an RNG so that repeated
    runs over the same corpus give byte-identical results -- a benchmark whose headline
    number moves between runs is not a benchmark.
    """

    name = "stub"

    def __init__(self, miss_rate: float = 0.12) -> None:
        self.miss_rate = miss_rate
        self.calls = 0

    def _hash_unit(self, *parts: str) -> float:
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") / 2**32

    def extract_field(self, text: str, field: str, doc_type: str) -> Optional[str]:
        self.calls += 1
        keywords = FIELD_KEYWORDS.get(field, [field.replace("_", " ")])
        hit = None
        for line in text.splitlines():
            low = line.lower()
            if any(kw in low for kw in keywords):
                hit = line
                break
        if hit is None:
            return None
        if self._hash_unit(field, hit) < self.miss_rate:
            return None  # simulated refusal / low-confidence abstention
        # Take everything after the last label separator on the line.
        tail = re.split(r"[:.]\s|\s{2,}|\.\s*\.", hit)[-1].strip(" .:")
        if not tail:
            return None
        if field in _NUMERIC_FIELDS:
            nums = re.findall(r"[\d][\d.,]*", tail)
            return nums[-1] if nums else None
        if field in _CODE_FIELDS:
            toks = re.findall(r"[A-Za-z]{2,3}\b", tail)
            return toks[0] if toks else None
        if field == "doc_date":
            m = re.search(r"[0-9A-Za-z]{1,4}[-/. ][0-9A-Za-z]{1,4}[-/. ][0-9]{2,4}", tail)
            return m.group(0) if m else None
        return tail


class AnthropicProvider:  # pragma: no cover - requires a key and network
    """Thin adapter. The SDK import happens inside the call, never at module import."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 64) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def extract_field(self, text: str, field: str, doc_type: str) -> Optional[str]:
        import anthropic

        client = anthropic.Anthropic()
        prompt = (
            f"Extract the field '{field}' from this {doc_type.replace('_', ' ')}.\n"
            "Reply with the raw value only, or NONE if it is not present.\n\n"
            f"{text}"
        )
        msg = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        out = msg.content[0].text.strip()
        return None if out.upper() == "NONE" else out


class OpenAIProvider:  # pragma: no cover - requires a key and network
    name = "openai"

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model

    def extract_field(self, text: str, field: str, doc_type: str) -> Optional[str]:
        import openai

        client = openai.OpenAI()
        prompt = (
            f"Extract the field '{field}' from this {doc_type.replace('_', ' ')}. "
            f"Reply with the raw value only, or NONE.\n\n{text}"
        )
        resp = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
        )
        out = (resp.choices[0].message.content or "").strip()
        return None if out.upper() == "NONE" else out
