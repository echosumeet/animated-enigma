"""Guarded text-to-SQL execution.

Defence in depth, because any single layer here is defeatable:

1. Static checks on the string  - single statement, must start with SELECT/WITH,
   no comment tokens (comment smuggling), no forbidden keywords.
2. sqlite3 ``set_authorizer``  - the only layer that actually matters. It runs
   inside the SQL parser, so it sees the statement the engine sees, not the one
   a regex thought it saw. Anything that is not a read of an allowlisted
   table/column is denied at prepare time.
3. Mandatory LIMIT injection and a hard row cap on the cursor.
4. A read-only connection handle, so DDL/DML cannot commit even if it parsed.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

MAX_ROWS = 200
DEFAULT_LIMIT = 100

#: table -> readable columns. Anything absent is denied by the authorizer.
ALLOWED_SCHEMA: dict[str, set[str]] = {
    "dim_calendar": {"date_id", "date", "year", "month", "week", "day_of_week"},
    "dim_sku": {"sku_id", "sku_name", "category", "unit_cost", "abc_class"},
    "dim_location": {"location_id", "location_name", "region", "location_type"},
    "dim_supplier": {"supplier_id", "supplier_name", "country", "on_time_target"},
    "fact_orders": {"order_id", "date_id", "sku_id", "location_id", "qty_ordered", "qty_shipped"},
    "fact_shipments": {"shipment_id", "date_id", "sku_id", "supplier_id", "location_id",
                       "qty", "promised_lead_days", "actual_lead_days"},
    "fact_inventory_snapshot": {"snapshot_id", "date_id", "sku_id", "location_id",
                                "on_hand", "on_order", "safety_stock"},
}

ALLOWED_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "round", "abs", "coalesce", "cast",
    "length", "lower", "upper", "substr", "julianday", "date", "strftime", "ifnull",
}

FORBIDDEN = re.compile(
    r"\b(attach|detach|pragma|insert|update|delete|drop|alter|create|replace|"
    r"vacuum|reindex|analyze|trigger|load_extension|readfile|writefile)\b",
    re.IGNORECASE,
)
COMMENT = re.compile(r"--|/\*|\*/|#")


class SQLGuardError(Exception):
    """Raised when a statement is refused. The message is safe to show a user."""


@dataclass
class GuardResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False
    notes: list[str] = field(default_factory=list)


def schema_prompt() -> str:
    """The schema grounding block handed to the model. Nothing else is visible."""
    lines = ["Readable tables (SELECT only):"]
    for table, cols in ALLOWED_SCHEMA.items():
        lines.append(f"  {table}({', '.join(sorted(cols))})")
    lines.append("Joins: facts carry sku_id, location_id, date_id; fact_shipments also supplier_id.")
    return "\n".join(lines)


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ:
        cols = ALLOWED_SCHEMA.get(arg1 or "")
        if cols is None:
            return sqlite3.SQLITE_DENY
        # sqlite passes an empty column name for "table referenced, no column read".
        if arg2 and arg2 not in cols:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def static_check(sql: str) -> str:
    """Cheap pre-parser checks. Returns the normalised statement."""
    s = (sql or "").strip()
    if not s:
        raise SQLGuardError("blocked: empty statement")
    if COMMENT.search(s):
        raise SQLGuardError("blocked: SQL comments are not allowed (comment smuggling)")
    body = s.rstrip(";").strip()
    if ";" in body:
        raise SQLGuardError("blocked: only a single statement may be executed (stacked statements)")
    if not re.match(r"^(select|with)\b", body, re.IGNORECASE):
        raise SQLGuardError("blocked: only SELECT/WITH statements are allowed")
    hit = FORBIDDEN.search(body)
    if hit:
        raise SQLGuardError(f"blocked: forbidden keyword '{hit.group(0).upper()}'")
    return body


def inject_limit(sql: str, limit: int = DEFAULT_LIMIT) -> tuple[str, bool]:
    """Append a LIMIT when the statement has none; clamp it when it is too large."""
    m = re.search(r"\blimit\b\s+(\d+)\s*$", sql, re.IGNORECASE)
    if m is None:
        return f"{sql} LIMIT {limit}", True
    if int(m.group(1)) > MAX_ROWS:
        return re.sub(r"\blimit\b\s+\d+\s*$", f"LIMIT {MAX_ROWS}", sql, flags=re.IGNORECASE), True
    return sql, False


def run_guarded_sql(conn: sqlite3.Connection, sql: str, limit: int = DEFAULT_LIMIT) -> GuardResult:
    """Execute ``sql`` under every guardrail. Raises SQLGuardError when refused."""
    body = static_check(sql)
    final, injected = inject_limit(body, limit)
    notes = ["limit injected/clamped"] if injected else []
    conn.set_authorizer(_authorizer)
    try:
        cur = conn.execute(final)
        rows = cur.fetchmany(MAX_ROWS + 1)
        cols = [d[0] for d in (cur.description or [])]
    except sqlite3.DatabaseError as exc:
        msg = str(exc)
        if "not authorized" in msg.lower() or "prohibited" in msg.lower():
            raise SQLGuardError("blocked: statement touches a table, column or function "
                                "outside the allowlist") from exc
        raise SQLGuardError(f"blocked: sqlite rejected the statement ({msg})") from exc
    finally:
        conn.set_authorizer(None)
    truncated = len(rows) > MAX_ROWS
    if truncated:
        rows = rows[:MAX_ROWS]
        notes.append(f"row cap {MAX_ROWS} applied")
    return GuardResult(sql=final, columns=cols, rows=[tuple(r) for r in rows],
                       truncated=truncated, notes=notes)
