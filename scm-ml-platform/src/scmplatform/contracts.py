"""Declarative data contracts: schema, expectations, validation, change detection.

A contract is the interface between the team that produces a planning feed and the
team that consumes it. Validation answers "is today's extract usable"; contract
diffing answers the question that actually causes outages -- "did the producer change
the interface in a way that silently breaks every downstream model".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd

Severity = Literal["error", "warn"]


@dataclass(frozen=True)
class ColumnSpec:
    """Expectations attached to one column."""

    name: str
    dtype: Literal["float", "int", "str", "bool", "datetime"]
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed: tuple[str, ...] | None = None
    severity: Severity = "error"

    def numpy_ok(self, series: pd.Series) -> bool:
        kind = series.dtype.kind
        return {
            "float": kind in "fiu",
            "int": kind in "iu",
            "str": kind in "OU",
            "bool": kind == "b",
            "datetime": kind == "M",
        }[self.dtype]


@dataclass(frozen=True)
class DataContract:
    """A named, versioned schema plus table-level expectations."""

    name: str
    version: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = ()
    freshness_column: str | None = None
    max_age_days: float | None = None

    def column(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)


@dataclass(frozen=True)
class Violation:
    column: str
    rule: str
    count: int
    detail: str
    severity: Severity = "error"


@dataclass
class ValidationReport:
    contract: str
    version: str
    rows: int
    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "column": v.column,
                    "rule": v.rule,
                    "count": v.count,
                    "severity": v.severity,
                    "detail": v.detail,
                }
                for v in self.violations
            ],
            columns=["column", "rule", "count", "severity", "detail"],
        )

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{status} {self.contract}@{self.version}: {self.rows} rows, "
            f"{len(self.errors)} error rule(s), "
            f"{len(self.violations) - len(self.errors)} warning rule(s)"
        )


def validate(
    df: pd.DataFrame, contract: DataContract, now: pd.Timestamp | None = None
) -> ValidationReport:
    """Check ``df`` against ``contract`` and return every rule that fired."""
    report = ValidationReport(contract.name, contract.version, len(df))
    add = report.violations.append

    for spec in contract.columns:
        if spec.name not in df.columns:
            add(Violation(spec.name, "missing_column", len(df), "column absent", "error"))
            continue
        s = df[spec.name]
        if not spec.numpy_ok(s):
            add(Violation(spec.name, "dtype", len(df), f"expected {spec.dtype}, got {s.dtype}", spec.severity))
        n_null = int(s.isna().sum())
        if not spec.nullable and n_null:
            add(Violation(spec.name, "not_null", n_null, f"{n_null} null values", spec.severity))
        if spec.min_value is not None and s.dtype.kind in "fiu":
            bad = int((s.dropna() < spec.min_value).sum())
            if bad:
                add(Violation(spec.name, "min_value", bad, f"{bad} rows < {spec.min_value}", spec.severity))
        if spec.max_value is not None and s.dtype.kind in "fiu":
            bad = int((s.dropna() > spec.max_value).sum())
            if bad:
                add(Violation(spec.name, "max_value", bad, f"{bad} rows > {spec.max_value}", spec.severity))
        if spec.allowed is not None:
            bad = int((~s.isin(spec.allowed) & s.notna()).sum())
            if bad:
                add(Violation(spec.name, "allowed_values", bad, f"{bad} rows outside {spec.allowed}", spec.severity))

    if contract.primary_key:
        present = [c for c in contract.primary_key if c in df.columns]
        if len(present) == len(contract.primary_key):
            dup = int(df.duplicated(subset=list(contract.primary_key)).sum())
            if dup:
                add(Violation("+".join(contract.primary_key), "unique_key", dup, f"{dup} duplicate keys", "error"))

    if contract.freshness_column and contract.max_age_days is not None:
        col = contract.freshness_column
        if col in df.columns and len(df):
            now = now if now is not None else pd.Timestamp.utcnow().tz_localize(None)
            age = (now - pd.to_datetime(df[col]).max()).total_seconds() / 86400.0
            if age > contract.max_age_days:
                add(Violation(col, "freshness", 1, f"max timestamp is {age:.2f} days old", "error"))
    return report


# --------------------------------------------------------------------------- diffing


@dataclass
class ContractDiff:
    breaking: list[str] = field(default_factory=list)
    non_breaking: list[str] = field(default_factory=list)

    @property
    def is_breaking(self) -> bool:
        return bool(self.breaking)

    def summary(self) -> str:
        verdict = "BREAKING" if self.is_breaking else "compatible"
        return f"{verdict}: {len(self.breaking)} breaking, {len(self.non_breaking)} non-breaking"


def _tighter(old: float | None, new: float | None, direction: int) -> bool:
    """True if the bound moved in the direction that rejects previously-valid rows."""
    if new is None:
        return False
    if old is None:
        return True
    return (new > old) if direction > 0 else (new < old)


def diff_contracts(old: DataContract, new: DataContract) -> ContractDiff:
    """Classify the change from ``old`` to ``new`` as breaking or non-breaking.

    Breaking means an existing, contract-conformant producer or consumer can now fail:
    dropped columns, dtype changes, newly-required columns, tightened bounds or value
    sets, a new uniqueness key, or a stricter freshness SLA.
    """
    d = ContractDiff()
    old_names = {c.name for c in old.columns}
    new_names = {c.name for c in new.columns}

    for name in sorted(old_names - new_names):
        d.breaking.append(f"column removed: {name}")
    for name in sorted(new_names - old_names):
        spec = new.column(name)
        (d.breaking if not spec.nullable else d.non_breaking).append(
            f"column added: {name}" + ("" if spec.nullable else " (non-nullable, requires backfill)")
        )

    for name in sorted(old_names & new_names):
        o, n = old.column(name), new.column(name)
        if o.dtype != n.dtype:
            d.breaking.append(f"dtype changed: {name} {o.dtype} -> {n.dtype}")
        if o.nullable and not n.nullable:
            d.breaking.append(f"nullability tightened: {name} now required")
        elif not o.nullable and n.nullable:
            d.non_breaking.append(f"nullability relaxed: {name} now optional")
        if _tighter(o.min_value, n.min_value, +1):
            d.breaking.append(f"min_value tightened: {name} {o.min_value} -> {n.min_value}")
        elif o.min_value is not None and (n.min_value is None or n.min_value < o.min_value):
            d.non_breaking.append(f"min_value relaxed: {name}")
        if _tighter(o.max_value, n.max_value, -1):
            d.breaking.append(f"max_value tightened: {name} {o.max_value} -> {n.max_value}")
        elif o.max_value is not None and (n.max_value is None or n.max_value > o.max_value):
            d.non_breaking.append(f"max_value relaxed: {name}")
        if o.allowed and n.allowed and set(n.allowed) < set(o.allowed):
            d.breaking.append(f"allowed values narrowed: {name}")
        elif o.allowed and n.allowed and set(n.allowed) > set(o.allowed):
            d.non_breaking.append(f"allowed values widened: {name}")

    if set(new.primary_key) and set(new.primary_key) != set(old.primary_key):
        d.breaking.append(f"primary key changed: {old.primary_key} -> {new.primary_key}")
    if new.max_age_days is not None and (
        old.max_age_days is None or new.max_age_days < old.max_age_days
    ):
        d.breaking.append(f"freshness SLA tightened: {old.max_age_days} -> {new.max_age_days} days")
    return d


def demand_panel_contract(version: str = "1.0.0") -> DataContract:
    """The contract the rest of this repository validates the demand feed against."""
    return DataContract(
        name="demand_panel",
        version=version,
        columns=(
            ColumnSpec("sku", "str", nullable=False),
            ColumnSpec("date", "datetime", nullable=False),
            ColumnSpec("region", "str", nullable=False, allowed=("north", "south", "east", "west")),
            ColumnSpec("category", "str", nullable=False),
            ColumnSpec("units", "float", nullable=False, min_value=0.0),
            ColumnSpec("price", "float", nullable=False, min_value=0.01, max_value=500.0),
            ColumnSpec("promo_flag", "int", nullable=False, min_value=0, max_value=1),
            ColumnSpec("on_hand", "float", nullable=True, min_value=0.0, severity="warn"),
        ),
        primary_key=("sku", "date"),
        freshness_column="date",
        max_age_days=2.0,
    )


def column_names(specs: Iterable[ColumnSpec]) -> list[str]:
    return [s.name for s in specs]


__all__ = [
    "ColumnSpec",
    "DataContract",
    "Violation",
    "ValidationReport",
    "ContractDiff",
    "validate",
    "diff_contracts",
    "demand_panel_contract",
    "column_names",
]
