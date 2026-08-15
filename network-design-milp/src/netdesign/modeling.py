"""A small algebraic modelling layer over ``scipy.optimize.milp``.

Why this exists: ``scipy.optimize.milp`` takes a dense objective vector, an
integrality mask, bounds, and a matrix of linear constraints. Writing a
multi-echelon network design model directly against that interface means
hand-maintaining an integer offset for every variable block, and every model
change becomes an index-arithmetic bug hunt. Those bugs are silent - the model
still solves, it just solves the wrong problem.

So this module provides the four things that make a MILP readable:

* a **variable registry** (:class:`Model.add_vars`) that keys variables by
  meaningful tuples such as ``("PLT02", "DC05", "rail", "SKU-A")`` rather than
  by position;
* **wildcard selection** (:meth:`VarGroup.sum`) so a flow-balance constraint
  reads like the algebra it came from;
* an **expression type** (:class:`LinExpr`) with the usual operators, so
  constraints are written with ``<=`` / ``==`` / ``>=``;
* **sparse assembly** into COO triplets, built once at solve time.

Everything is deliberately thin. There is no presolve, no cut generation, no
column generation - HiGHS does that. The goal is only that the model in the
source code looks like the model on the whiteboard.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

__all__ = [
    "ANY",
    "LinExpr",
    "RowConstraint",
    "Model",
    "Solution",
    "VarGroup",
    "quicksum",
    "SolverStatus",
]

#: Wildcard token accepted by :meth:`VarGroup.select` and :meth:`VarGroup.sum`.
ANY = "*"

_INF = float("inf")

Key = Hashable


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------


class LinExpr:
    """A sparse linear expression ``sum(c_j * x_j) + constant``.

    Coefficients are stored in a dict keyed by the model's internal column
    index. Arithmetic always returns a new expression; nothing mutates in
    place, which keeps the accidental-aliasing class of bug out of the model
    code.
    """

    __slots__ = ("coeffs", "constant")

    def __init__(
        self,
        coeffs: Mapping[int, float] | None = None,
        constant: float = 0.0,
    ) -> None:
        self.coeffs: dict[int, float] = dict(coeffs) if coeffs else {}
        self.constant = float(constant)

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_column(cls, index: int, coef: float = 1.0) -> LinExpr:
        return cls({index: float(coef)})

    def copy(self) -> LinExpr:
        return LinExpr(self.coeffs, self.constant)

    # -- arithmetic ----------------------------------------------------------

    def _combine(self, other: Any, sign: float) -> LinExpr:
        out = self.copy()
        if isinstance(other, LinExpr):
            for j, c in other.coeffs.items():
                new = out.coeffs.get(j, 0.0) + sign * c
                if new == 0.0:
                    out.coeffs.pop(j, None)
                else:
                    out.coeffs[j] = new
            out.constant += sign * other.constant
        elif isinstance(other, (int, float, np.integer, np.floating)):
            out.constant += sign * float(other)
        else:  # pragma: no cover - guard
            return NotImplemented
        return out

    def __add__(self, other: Any) -> LinExpr:
        return self._combine(other, 1.0)

    __radd__ = __add__

    def __sub__(self, other: Any) -> LinExpr:
        return self._combine(other, -1.0)

    def __rsub__(self, other: Any) -> LinExpr:
        return (-self)._combine(other, 1.0)

    def __neg__(self) -> LinExpr:
        return LinExpr({j: -c for j, c in self.coeffs.items()}, -self.constant)

    def __mul__(self, scalar: Any) -> LinExpr:
        if not isinstance(scalar, (int, float, np.integer, np.floating)):
            raise TypeError(
                "LinExpr can only be multiplied by a scalar; "
                "products of two variables are not linear."
            )
        s = float(scalar)
        if s == 0.0:
            return LinExpr()
        return LinExpr({j: c * s for j, c in self.coeffs.items()}, self.constant * s)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> LinExpr:
        return self.__mul__(1.0 / float(scalar))

    # -- comparisons build constraints --------------------------------------

    def __le__(self, other: Any) -> RowConstraint:
        diff = self - other
        return RowConstraint(diff.coeffs, -_INF, -diff.constant)

    def __ge__(self, other: Any) -> RowConstraint:
        diff = self - other
        return RowConstraint(diff.coeffs, -diff.constant, _INF)

    def __eq__(self, other: Any) -> RowConstraint:  # type: ignore[override]
        diff = self - other
        return RowConstraint(diff.coeffs, -diff.constant, -diff.constant)

    __hash__ = None  # type: ignore[assignment]

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> float:
        total = self.constant
        for j, c in self.coeffs.items():
            total += c * float(x[j])
        return total

    def __len__(self) -> int:
        return len(self.coeffs)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"LinExpr(nnz={len(self.coeffs)}, const={self.constant:g})"


def quicksum(terms: Iterable[Any]) -> LinExpr:
    """Sum an iterable of expressions/scalars in O(nnz) without temporaries."""
    out = LinExpr()
    for t in terms:
        if isinstance(t, LinExpr):
            for j, c in t.coeffs.items():
                new = out.coeffs.get(j, 0.0) + c
                if new == 0.0:
                    out.coeffs.pop(j, None)
                else:
                    out.coeffs[j] = new
            out.constant += t.constant
        else:
            out.constant += float(t)
    return out


@dataclass
class RowConstraint:
    """A two-sided linear row ``lb <= a'x <= ub``."""

    coeffs: dict[int, float]
    lb: float
    ub: float
    name: str = ""
    tag: str = ""

    def with_meta(self, name: str, tag: str) -> RowConstraint:
        self.name = name
        self.tag = tag
        return self


# ---------------------------------------------------------------------------
# variable groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarSpec:
    """Everything the assembler needs about one column."""

    index: int
    name: str
    lb: float
    ub: float
    integrality: int  # 0 continuous, 1 integer/binary


class VarGroup(Mapping[Key, LinExpr]):
    """A named family of variables indexed by hashable keys.

    Keys are normally tuples, and tuple keys get a positional inverted index so
    that ``group.sum(ANY, dc, ANY, sku)`` is a set intersection rather than a
    scan over every variable in the group. In a network design model the
    balance constraints are written exactly that way, so the difference matters
    once the arc set is in the tens of thousands.
    """

    __slots__ = ("name", "_exprs", "_order", "_index", "_arity")

    def __init__(self, name: str) -> None:
        self.name = name
        self._exprs: dict[Key, LinExpr] = {}
        self._order: dict[Key, int] = {}
        self._index: list[dict[Any, set[Key]]] = []
        self._arity: int | None = None

    def _register(self, key: Key, expr: LinExpr) -> None:
        if key in self._exprs:
            raise KeyError(f"duplicate key {key!r} in variable group {self.name!r}")
        self._order[key] = len(self._exprs)
        self._exprs[key] = expr
        if isinstance(key, tuple):
            if self._arity is None:
                self._arity = len(key)
                self._index = [dict() for _ in range(len(key))]
            if len(key) == self._arity:
                for pos, part in enumerate(key):
                    self._index[pos].setdefault(part, set()).add(key)

    # -- Mapping protocol ----------------------------------------------------

    def __getitem__(self, key: Key) -> LinExpr:
        return self._exprs[key]

    def __iter__(self) -> Iterator[Key]:
        return iter(self._exprs)

    def __len__(self) -> int:
        return len(self._exprs)

    # -- selection -----------------------------------------------------------

    def select(self, *pattern: Any) -> list[Key]:
        """Return keys matching ``pattern``; :data:`ANY` matches any value."""
        if not pattern:
            return list(self._exprs)
        if self._arity is None or len(pattern) != self._arity:
            raise ValueError(
                f"group {self.name!r} has key arity {self._arity}, "
                f"got pattern of length {len(pattern)}"
            )
        fixed = [(pos, val) for pos, val in enumerate(pattern) if val is not ANY]
        if not fixed:
            return list(self._exprs)
        # start from the most selective position to keep intersections small
        fixed.sort(key=lambda pv: len(self._index[pv[0]].get(pv[1], ())))
        pos, val = fixed[0]
        matched = set(self._index[pos].get(val, ()))
        for pos, val in fixed[1:]:
            if not matched:
                break
            matched &= self._index[pos].get(val, set())
        return sorted(matched, key=self._order.__getitem__)

    def sum(self, *pattern: Any) -> LinExpr:
        """Sum of the variables matching ``pattern`` (empty sum is ``0``)."""
        return quicksum(self._exprs[k] for k in self.select(*pattern))

    def sum_over(self, keys: Iterable[Key]) -> LinExpr:
        return quicksum(self._exprs[k] for k in keys if k in self._exprs)

    def dot(self, weights: Mapping[Key, float]) -> LinExpr:
        """Weighted sum ``sum(w_k * x_k)`` over the keys present in *weights*."""
        out = LinExpr()
        for k, w in weights.items():
            expr = self._exprs.get(k)
            if expr is None or w == 0.0:
                continue
            for j, c in expr.coeffs.items():
                out.coeffs[j] = out.coeffs.get(j, 0.0) + c * float(w)
        return out

    def columns(self) -> dict[Key, int]:
        """Map key -> column index (single-column variables only)."""
        out: dict[Key, int] = {}
        for k, e in self._exprs.items():
            (j,) = e.coeffs
            out[k] = j
        return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class SolverStatus:
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    ERROR = "error"


_STATUS_MAP = {
    0: SolverStatus.OPTIMAL,
    1: SolverStatus.ITERATION_LIMIT,
    2: SolverStatus.INFEASIBLE,
    3: SolverStatus.UNBOUNDED,
    4: SolverStatus.ERROR,
}


@dataclass
class Solution:
    """The result of solving a :class:`Model`."""

    status: str
    objective: float | None
    x: np.ndarray | None
    runtime: float
    mip_gap: float | None = None
    message: str = ""
    relaxed: bool = False

    @property
    def is_optimal(self) -> bool:
        return self.status == SolverStatus.OPTIMAL

    def require_optimal(self) -> Solution:
        if not self.is_optimal:
            raise RuntimeError(f"solve did not reach optimality: {self.status} - {self.message}")
        return self

    def value(self, expr: LinExpr) -> float:
        if self.x is None:
            raise RuntimeError("no solution vector available")
        return expr.evaluate(self.x)

    def values(
        self,
        group: VarGroup,
        *,
        tol: float = 1e-7,
        nonzero: bool = False,
        integral: bool = False,
    ) -> dict[Key, float]:
        """Extract the solved value of every variable in *group*.

        ``integral=True`` rounds values that are within *tol* of an integer,
        which is what you want for binaries: HiGHS legitimately returns
        ``0.9999999997`` and downstream ``if y == 1`` then silently drops a
        facility from the report.
        """
        if self.x is None:
            raise RuntimeError("no solution vector available")
        out: dict[Key, float] = {}
        for k, expr in group.items():
            v = expr.evaluate(self.x)
            if integral and abs(v - round(v)) <= 1e-6:
                v = float(round(v))
            if abs(v) <= tol:
                if nonzero:
                    continue
                v = 0.0
            out[k] = v
        return out


class Model:
    """A linear/mixed-integer program built from named variable groups."""

    def __init__(self, name: str = "model", sense: str = "min") -> None:
        if sense not in ("min", "max"):
            raise ValueError("sense must be 'min' or 'max'")
        self.name = name
        self.sense = sense
        self._specs: list[VarSpec] = []
        self._rows: list[RowConstraint] = []
        self._groups: dict[str, VarGroup] = {}
        self._objective = LinExpr()

    # -- variables -----------------------------------------------------------

    def _new_column(self, name: str, lb: float, ub: float, integrality: int) -> int:
        j = len(self._specs)
        self._specs.append(VarSpec(j, name, float(lb), float(ub), integrality))
        return j

    def add_var(
        self,
        name: str,
        *,
        lb: float = 0.0,
        ub: float = _INF,
        vtype: str = "continuous",
        obj: float = 0.0,
    ) -> LinExpr:
        """Add a single column and return the expression that references it."""
        lb, ub, integrality = _resolve_type(vtype, lb, ub)
        j = self._new_column(name, lb, ub, integrality)
        if obj:
            self._objective = self._objective + obj * LinExpr.from_column(j)
        return LinExpr.from_column(j)

    def add_vars(
        self,
        keys: Iterable[Key],
        *,
        name: str,
        lb: float | Callable[[Key], float] = 0.0,
        ub: float | Callable[[Key], float] = _INF,
        vtype: str = "continuous",
        obj: float | Callable[[Key], float] | Mapping[Key, float] | None = None,
    ) -> VarGroup:
        """Add a family of columns keyed by *keys*.

        ``lb``/``ub``/``obj`` accept a scalar, a callable of the key, or (for
        ``obj``) a mapping. Objective coefficients supplied here are added to
        the model objective directly, which avoids a second pass over the arc
        set purely to build the cost vector.
        """
        if name in self._groups:
            raise KeyError(f"variable group {name!r} already exists")
        group = VarGroup(name)
        obj_terms: dict[int, float] = {}
        for key in keys:
            klb = lb(key) if callable(lb) else lb
            kub = ub(key) if callable(ub) else ub
            klb, kub, integrality = _resolve_type(vtype, klb, kub)
            j = self._new_column(f"{name}[{_fmt_key(key)}]", klb, kub, integrality)
            group._register(key, LinExpr.from_column(j))
            if obj is not None:
                if callable(obj):
                    c = float(obj(key))
                elif isinstance(obj, Mapping):
                    c = float(obj.get(key, 0.0))
                else:
                    c = float(obj)
                if c:
                    obj_terms[j] = c
        if obj_terms:
            self._objective = self._objective + LinExpr(obj_terms)
        self._groups[name] = group
        return group

    def group(self, name: str) -> VarGroup:
        return self._groups[name]

    # -- constraints ---------------------------------------------------------

    def add(self, constraint: RowConstraint, *, name: str = "", tag: str = "") -> RowConstraint:
        if not isinstance(constraint, RowConstraint):
            raise TypeError(
                "add() expects a comparison of expressions, e.g. "
                "model.add(flow.sum(ANY, d) <= cap)"
            )
        if not constraint.coeffs:
            # A constant row: either trivially true or a modelling error worth
            # surfacing now rather than as a mysterious infeasibility later.
            if constraint.lb > 1e-9 or constraint.ub < -1e-9:
                raise ValueError(
                    f"constraint {name or '<unnamed>'} has no variables and cannot hold "
                    f"({constraint.lb} <= 0 <= {constraint.ub})"
                )
            return constraint
        constraint.with_meta(name, tag)
        self._rows.append(constraint)
        return constraint

    def add_all(
        self, constraints: Iterable[RowConstraint], *, name: str = "", tag: str = ""
    ) -> None:
        for i, c in enumerate(constraints):
            self.add(c, name=f"{name}[{i}]" if name else "", tag=tag)

    def link_big_m(
        self,
        activity: LinExpr,
        indicator: LinExpr,
        big_m: float,
        *,
        name: str = "",
        tag: str = "bigM",
    ) -> RowConstraint:
        """Add ``activity <= big_m * indicator``.

        The helper exists to make the modeller pass ``big_m`` explicitly. The
        classic defect is a shared "large number" constant: the formulation is
        valid but the LP relaxation is worthless, and branch and bound then
        spends its life proving bounds it should have got for free. See
        :func:`netdesign.network_flow.throughput_big_m` for how the tight value
        is derived here.
        """
        if not math.isfinite(big_m) or big_m <= 0:
            raise ValueError(f"big_m must be a positive finite number, got {big_m!r}")
        return self.add(activity - big_m * indicator <= 0.0, name=name, tag=tag)

    # -- objective -----------------------------------------------------------

    def set_objective(self, expr: LinExpr, sense: str | None = None) -> None:
        self._objective = expr if isinstance(expr, LinExpr) else LinExpr(constant=float(expr))
        if sense is not None:
            self.sense = sense

    def add_objective(self, expr: LinExpr) -> None:
        self._objective = self._objective + expr

    @property
    def objective(self) -> LinExpr:
        return self._objective

    # -- introspection -------------------------------------------------------

    @property
    def num_vars(self) -> int:
        return len(self._specs)

    @property
    def num_integer_vars(self) -> int:
        return sum(s.integrality for s in self._specs)

    @property
    def num_constraints(self) -> int:
        return len(self._rows)

    @property
    def num_nonzeros(self) -> int:
        return sum(len(r.coeffs) for r in self._rows)

    def rows_by_tag(self, tag: str) -> list[RowConstraint]:
        return [r for r in self._rows if r.tag == tag]

    def tags(self) -> list[str]:
        """Every distinct constraint tag in the model, in first-seen order."""
        seen: dict[str, None] = {}
        for r in self._rows:
            if r.tag:
                seen.setdefault(r.tag, None)
        return list(seen)

    def stats(self) -> dict[str, int]:
        return {
            "variables": self.num_vars,
            "binaries": self.num_integer_vars,
            "constraints": self.num_constraints,
            "nonzeros": self.num_nonzeros,
        }

    # -- assembly ------------------------------------------------------------

    def assemble(self, *, relax: bool = False) -> dict[str, Any]:
        """Build the dense/sparse arrays ``scipy.optimize.milp`` consumes."""
        n = len(self._specs)
        if n == 0:
            raise ValueError("model has no variables")
        c = np.zeros(n)
        for j, coef in self._objective.coeffs.items():
            c[j] = coef
        if self.sense == "max":
            c = -c
        lb = np.fromiter((s.lb for s in self._specs), dtype=float, count=n)
        ub = np.fromiter((s.ub for s in self._specs), dtype=float, count=n)
        integrality = np.fromiter(
            ((0 if relax else s.integrality) for s in self._specs), dtype=int, count=n
        )

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for i, row in enumerate(self._rows):
            for j, coef in row.coeffs.items():
                rows.append(i)
                cols.append(j)
                data.append(coef)
        m = len(self._rows)
        A = sparse.coo_matrix(
            (np.asarray(data, dtype=float), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
            shape=(m, n),
        ).tocsr()
        clb = np.fromiter((r.lb for r in self._rows), dtype=float, count=m)
        cub = np.fromiter((r.ub for r in self._rows), dtype=float, count=m)
        return {
            "c": c,
            "A": A,
            "constraint_lb": clb,
            "constraint_ub": cub,
            "lb": lb,
            "ub": ub,
            "integrality": integrality,
        }

    # -- solve ---------------------------------------------------------------

    def solve(
        self,
        *,
        relax: bool = False,
        time_limit: float | None = None,
        mip_rel_gap: float = 1e-6,
        presolve: bool = True,
        verbose: bool = False,
    ) -> Solution:
        """Solve with HiGHS via ``scipy.optimize.milp``.

        ``relax=True`` drops integrality, which is how the LP bound reported in
        the benchmarks is obtained - the single most useful number for deciding
        whether a formulation is worth keeping.
        """
        parts = self.assemble(relax=relax)
        constraints = (
            [LinearConstraint(parts["A"], parts["constraint_lb"], parts["constraint_ub"])]
            if parts["A"].shape[0] > 0
            else []
        )
        options: dict[str, Any] = {
            "presolve": presolve,
            "disp": verbose,
            "mip_rel_gap": mip_rel_gap,
        }
        if time_limit is not None:
            options["time_limit"] = float(time_limit)

        t0 = time.perf_counter()
        res = milp(
            c=parts["c"],
            constraints=constraints,
            integrality=parts["integrality"],
            bounds=Bounds(parts["lb"], parts["ub"]),
            options=options,
        )
        runtime = time.perf_counter() - t0

        status = _STATUS_MAP.get(int(res.status), SolverStatus.ERROR)
        if status == SolverStatus.ITERATION_LIMIT and time_limit is not None:
            status = SolverStatus.TIME_LIMIT
        obj = None
        x = None
        if res.x is not None:
            x = np.asarray(res.x, dtype=float)
            obj = self._objective.evaluate(x)
        gap = getattr(res, "mip_gap", None)
        return Solution(
            status=status,
            objective=obj,
            x=x,
            runtime=runtime,
            mip_gap=float(gap) if gap is not None else None,
            message=str(res.message),
            relaxed=relax,
        )

    # -- elastic relaxation (feasibility diagnostics) ------------------------

    def elastic_copy(
        self,
        tags: Iterable[str],
        *,
        penalty: float = 1.0,
        keep_objective_weight: float = 0.0,
    ) -> tuple[Model, dict[int, RowConstraint], VarGroup]:
        """Return a copy in which rows carrying *tags* may be violated at a cost.

        This is the practical stand-in for an IIS: minimise total violation and
        read off which constraint groups had to give, and by how much. It
        answers the only question a planner asks about an infeasible network -
        *what do I have to change* - which "model status: infeasible" does not.
        """
        tagset = set(tags)
        clone = Model(name=f"{self.name}:elastic", sense="min")
        clone._specs = list(self._specs)
        clone._groups = dict(self._groups)
        clone._rows = [
            RowConstraint(dict(r.coeffs), r.lb, r.ub, r.name, r.tag) for r in self._rows
        ]

        elastic_rows: dict[int, RowConstraint] = {}
        slack_keys: list[Key] = []
        for i, row in enumerate(clone._rows):
            if row.tag in tagset:
                elastic_rows[i] = row
                if row.lb > -_INF:
                    slack_keys.append((i, "under"))
                if row.ub < _INF:
                    slack_keys.append((i, "over"))
        slacks = VarGroup("violation")
        for key in slack_keys:
            i, side = key  # type: ignore[misc]
            j = clone._new_column(f"violation[{i},{side}]", 0.0, _INF, 0)
            slacks._register(key, LinExpr.from_column(j))
            row = clone._rows[i]
            # +s relaxes a >= row, -s relaxes a <= row
            row.coeffs[j] = 1.0 if side == "under" else -1.0
        clone._groups["violation"] = slacks

        obj = penalty * quicksum(slacks.values())
        if keep_objective_weight:
            obj = obj + keep_objective_weight * self._objective
        clone.set_objective(obj, "min")
        return clone, elastic_rows, slacks

    def fix(self, expr: LinExpr, value: float, *, name: str = "", tag: str = "fix") -> None:
        """Pin an expression (usually a single binary) to a value."""
        self.add(expr == float(value), name=name, tag=tag)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        s = self.stats()
        return (
            f"Model({self.name!r}, vars={s['variables']}, int={s['binaries']}, "
            f"rows={s['constraints']}, nnz={s['nonzeros']})"
        )


def _resolve_type(vtype: str, lb: float, ub: float) -> tuple[float, float, int]:
    vtype = vtype.lower()
    if vtype in ("c", "continuous"):
        return float(lb), float(ub), 0
    if vtype in ("b", "binary"):
        return 0.0, 1.0, 1
    if vtype in ("i", "integer"):
        return float(lb), float(ub), 1
    raise ValueError(f"unknown variable type {vtype!r}")


def _fmt_key(key: Key) -> str:
    if isinstance(key, tuple):
        return ",".join(str(k) for k in key)
    return str(key)
