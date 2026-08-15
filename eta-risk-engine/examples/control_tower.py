"""One shift in a control tower.

Trains on history, scores the next block of bookings, and prints what an
exception desk would actually see: the ten shipments with the highest delay
risk, the ETA interval you would put in front of a customer, and what the cost
matrix says to do about each one.

    PYTHONPATH=src python examples/control_tower.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from etarisk.decision import ACTIONS, CostMatrix, optimal_actions  # noqa: E402
from etarisk.drift import psi_table  # noqa: E402
from etarisk.generate import GeneratorConfig  # noqa: E402
from etarisk.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    pd.set_option("display.width", 150)
    costs = CostMatrix()
    res = run_pipeline(GeneratorConfig(n_shipments=30_000), alphas=(0.1,), costs=costs, max_iter=200)
    m = res.metrics
    test, X_test = res.splits["test"], res.X["test"]

    point, lo, hi = res.conformal[0.1].predict_interval(X_test, test["planned_transit_h"].to_numpy())
    p_late = res.risk.predict_proba(X_test)
    actions = optimal_actions(p_late, costs)

    print("=== Panel " + "=" * 62)
    print(
        f"{m['n_train']:,} training / {m['n_calib']:,} calibration / {m['n_test']:,} scored shipments; "
        f"realised late rate {m['late_rate_test']:.1%}"
    )
    print(f"point ETA MAE {m['eta_mae_h']:.2f} h vs quoted transit time {m['quoted_mae_h']:.2f} h")
    cov = m["coverage"].iloc[0]
    print(
        f"90% conformal interval: empirical coverage {cov['empirical']:.3f}, "
        f"median width {cov['median_width_h']:.1f} h"
    )

    print("\n=== Today's exception list (highest delay risk) " + "=" * 26)
    board = pd.DataFrame(
        {
            "ship_date": test["ship_ts"].dt.date.to_numpy(),
            "lane": test["lane"].to_numpy(),
            "mode": test["mode"].to_numpy(),
            "carrier": test["carrier"].to_numpy(),
            "quoted_h": test["planned_transit_h"].to_numpy(),
            "eta_h": np.round(point, 1),
            "eta_lo": np.round(lo, 1),
            "eta_hi": np.round(hi, 1),
            "p_late": np.round(p_late, 3),
            "action": [ACTIONS[i] for i in actions],
            "actual_h": test["actual_transit_h"].to_numpy(),
        }
    )
    print(board.sort_values("p_late", ascending=False).head(10).to_string(index=False))

    print("\n=== Action mix and cost " + "=" * 50)
    d = m["decision"]
    mix = pd.Series([ACTIONS[i] for i in actions]).value_counts(normalize=True).sort_index()
    print(mix.map("{:.1%}".format).to_string())
    print(
        f"cost/shipment: model {d['model_cost_per_shipment']:.2f}, "
        f"fixed carrier-scorecard rule {d['baseline_cost_per_shipment']:.2f}, "
        f"oracle {d['oracle_cost_per_shipment']:.2f}"
    )
    print(
        f"saved vs fixed rule: {d['saved_per_shipment']:.2f} per shipment "
        f"({d['saved_pct']:.1f}%), i.e. {d['pct_of_oracle_gap_captured']:.1f}% of the oracle gap"
    )

    print("\n=== Drift check, training block vs scored block " + "=" * 26)
    print(psi_table(res.X["train_scoring"], X_test).head(5).to_string(index=False))
    print("\nAnything above 0.25 means the encoder is being asked about a network it has not seen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
