"""Command line entry point: python -m sccopilot <ask|eval|attacks|schema|figures>."""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .evals import paraphrased_set, run_attack_suite, run_eval
from .guard import schema_prompt
from .warehouse import build_warehouse, row_counts


def _figure(outdir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conn = build_warehouse()
    report = run_eval(conn)
    para = run_eval(conn, items=paraphrased_set())
    cats, pcats = report["by_category"], para["by_category"]
    order = ["lookup", "aggregation", "multi_hop", "what_if", "ambiguous", "refusal", "adversarial"]
    names = [c for c in order if c in cats]
    series = [("golden phrasing", [cats[c]["pass_rate"] for c in names], "#2f5d8a"),
              ("paraphrased", [pcats[c]["pass_rate"] for c in names], "#8fb0cc"),
              ("tool selection (golden)", [cats[c]["tool_accuracy"] for c in names], "#c9c9c9")]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = range(len(names))
    for k, (label, vals, colour) in enumerate(series):
        off = (k - 1) * 0.27
        ax.bar([i + off for i in x], vals, width=0.26, color=colour, label=label)
        for i, v in enumerate(vals):
            ax.text(i + off, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace("_", " ") for n in names], fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("rate")
    ax.set_title(f"Golden-set eval by category (n={report['n']} per arm, stub provider)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = f"{outdir}/eval_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sccopilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="ask the copilot one question")
    a.add_argument("question", nargs="+")
    a.add_argument("--trace", action="store_true")
    sub.add_parser("eval", help="run the golden-set eval")
    sub.add_parser("attacks", help="run the guardrail attack suite")
    sub.add_parser("schema", help="print the grounded schema and row counts")
    f = sub.add_parser("figures", help="regenerate the eval figure")
    f.add_argument("--outdir", default="docs")
    args = p.parse_args(argv)

    if args.cmd == "figures":
        print(_figure(args.outdir))
        return 0

    conn = build_warehouse()
    if args.cmd == "ask":
        res = Agent(conn).run(" ".join(args.question))
        print(res.answer)
        if args.trace:
            print(res.trace_json())
    elif args.cmd == "eval":
        rep = run_eval(conn)
        par = run_eval(conn, items=paraphrased_set())
        print(f"n={rep['n']} pass_rate={rep['pass_rate']:.3f} "
              f"tool_accuracy={rep['tool_accuracy']:.3f} avg_steps={rep['avg_steps']}")
        print(f"paraphrase arm: pass_rate={par['pass_rate']:.3f} "
              f"tool_accuracy={par['tool_accuracy']:.3f}")
        for cat, c in rep["by_category"].items():
            print(f"  {cat:<12} n={c['n']:<3} pass={c['pass_rate']:.2f} "
                  f"tools={c['tool_accuracy']:.2f} steps={c['avg_steps']}")
        for s in rep["items"]:
            if not s.passed:
                print(f"  FAIL {s.id} ({s.category}) tools={s.tools_used} {s.detail}")
    elif args.cmd == "attacks":
        for row in run_attack_suite(conn):
            print(f"{'BLOCKED' if row['blocked'] else 'EXECUTED':<8} "
                  f"{row['attack']:<22} {row['layer']:<13} {row['reason']}")
    elif args.cmd == "schema":
        print(schema_prompt())
        print("\nrow counts:")
        for t, n in row_counts(conn).items():
            print(f"  {t:<26} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
