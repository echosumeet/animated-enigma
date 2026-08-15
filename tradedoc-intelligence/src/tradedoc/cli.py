"""Command line entry point: ``python -m tradedoc <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from .evaluate import run
from .generate import NoiseConfig, build_corpus, write_corpus


def _noise(args) -> NoiseConfig:
    return NoiseConfig(char_corruption_rate=args.corruption,
                       field_dropout_rate=args.dropout,
                       inconsistency_rate=args.inconsistency)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tradedoc", description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-n", "--shipments", type=int, default=120)
    ap.add_argument("--corruption", type=float, default=0.006)
    ap.add_argument("--dropout", type=float, default=0.07)
    ap.add_argument("--inconsistency", type=float, default=0.18)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a document corpus with ground truth")
    g.add_argument("--outdir", default="corpus")
    g.add_argument("--pdf", action="store_true", help="also emit reportlab PDFs")

    sub.add_parser("evaluate", help="run extraction, reconciliation and the sweep")

    f = sub.add_parser("figures", help="regenerate the STP curve")
    f.add_argument("--outdir", default="docs")

    args = ap.parse_args(argv)

    if args.cmd == "generate":
        corpus = build_corpus(args.shipments, args.seed, _noise(args))
        n = write_corpus(corpus, args.outdir, pdf=args.pdf)
        print(f"wrote {n} documents (+ ground truth) to {args.outdir}")
        return 0

    out = run(args.shipments, args.seed, _noise(args))
    if args.cmd == "figures":
        from .figures import stp_curve

        path = stp_curve(out, f"{args.outdir}/stp_curve.png")
        print(f"wrote {path}")
        return 0

    op = out.operating
    summary = {
        "documents": len(out.results),
        "document_level_accuracy": round(sum(out.correct) / len(out.correct), 4),
        "macro_field_accuracy": {k: v["macro_field_accuracy"] for k, v in out.accuracy.items()},
        "operating_point": None if op is None else {
            "threshold": op.threshold, "stp_rate": round(op.stp_rate, 4),
            "precision": op.precision, "recall": op.recall, "error_rate": op.error_rate,
        },
        "reconciliation": {k: out.detection[k] for k in ("precision", "recall")},
        "llm_calls": out.llm_calls,
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
