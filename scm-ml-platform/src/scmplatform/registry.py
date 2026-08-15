"""File-backed model registry: model cards, lineage, stage transitions, rollback.

Deliberately a directory of JSON files. A registry earns its keep through the
questions it can answer under pressure -- "what is serving right now", "what data and
feature set produced it", "what was serving before, and can I put it back in one
command" -- not through its storage engine. Everything here is auditable with ``cat``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("dev", "staging", "production", "archived")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ModelCard:
    """The record a reviewer reads before approving a promotion."""

    name: str
    version: str
    stage: str = "dev"
    created_at: str = field(default_factory=_now)
    intended_use: str = ""
    owner: str = ""
    algorithm: str = ""
    training_window: str = ""
    feature_specs: list[str] = field(default_factory=list)
    data_contract: str = ""
    dataset_fingerprint: str = ""
    parent_version: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    limitations: str = ""
    review_notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


class RegistryError(RuntimeError):
    pass


class ModelRegistry:
    """A registry rooted at a directory on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "models").mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / "history.jsonl"

    # ------------------------------------------------------------------ internals
    def _path(self, name: str, version: str) -> Path:
        return self.root / "models" / name / f"{version}.json"

    def _log(self, event: str, **payload) -> None:
        rec = {"ts": _now(), "event": event, **payload}
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # --------------------------------------------------------------------- writes
    def register(self, card: ModelCard, overwrite: bool = False) -> ModelCard:
        path = self._path(card.name, card.version)
        if path.exists() and not overwrite:
            raise RegistryError(f"{card.name}@{card.version} already registered")
        if card.stage not in STAGES:
            raise RegistryError(f"unknown stage {card.stage!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(card.to_json(), encoding="utf-8")
        self._log("register", name=card.name, version=card.version, stage=card.stage)
        return card

    def transition(self, name: str, version: str, stage: str, note: str = "") -> ModelCard:
        """Move a version to ``stage``; only one version is in production at a time."""
        if stage not in STAGES:
            raise RegistryError(f"unknown stage {stage!r}")
        card = self.get(name, version)
        previous = self.production(name) if stage == "production" else None
        if previous is not None and previous.version != version:
            previous.stage = "archived"
            self._path(name, previous.version).write_text(previous.to_json(), encoding="utf-8")
            self._log("archive", name=name, version=previous.version, reason="superseded")
        from_stage, card.stage = card.stage, stage
        if note:
            card.review_notes = note
        self._path(name, version).write_text(card.to_json(), encoding="utf-8")
        self._log("transition", name=name, version=version, **{"from": from_stage, "to": stage})
        return card

    def rollback(self, name: str, note: str = "") -> ModelCard:
        """Restore the version that was in production immediately before the current one."""
        target = self.previous_production(name)
        if target is None:
            raise RegistryError(f"no prior production version for {name}")
        self._log("rollback", name=name, to=target)
        return self.transition(name, target, "production", note=note or "rollback")

    # ---------------------------------------------------------------------- reads
    def get(self, name: str, version: str) -> ModelCard:
        path = self._path(name, version)
        if not path.exists():
            raise RegistryError(f"{name}@{version} not found")
        return ModelCard(**json.loads(path.read_text(encoding="utf-8")))

    def versions(self, name: str) -> list[str]:
        d = self.root / "models" / name
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []

    def production(self, name: str) -> ModelCard | None:
        for v in self.versions(name):
            card = self.get(name, v)
            if card.stage == "production":
                return card
        return None

    def history(self, name: str | None = None) -> list[dict]:
        if not self.history_path.exists():
            return []
        recs = [json.loads(line) for line in self.history_path.read_text().splitlines() if line]
        return [r for r in recs if name is None or r.get("name") == name]

    def previous_production(self, name: str) -> str | None:
        """The version promoted to production before the one promoted most recently."""
        promotions = [
            r["version"]
            for r in self.history(name)
            if r["event"] == "transition" and r.get("to") == "production"
        ]
        current = self.production(name)
        distinct: list[str] = []
        for v in promotions:
            if not distinct or distinct[-1] != v:
                distinct.append(v)
        if current is not None:
            distinct = [v for v in distinct if v != current.version]
        return distinct[-1] if distinct else None

    def lineage(self, name: str, version: str) -> list[ModelCard]:
        """Walk parent pointers from ``version`` back to the root model."""
        chain, seen = [], set()
        cur: str | None = version
        while cur and cur not in seen:
            seen.add(cur)
            card = self.get(name, cur)
            chain.append(card)
            cur = card.parent_version
        return chain
