"""End-to-end pipeline: generate data, fit models, optimize budget, price risk.

Writes:
  outputs/metrics.json   - every measured number (single source of truth)
  docs/assets/*.png      - rendered figures
  README.md              - generated from outputs/metrics.json (numbers match by construction)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import budget_optimization as bo
from src import figures as fig
from src import insurance_models as im
from src import marketing_models as mm
from src.config import (
    BUDGET_TOTAL,
    CHANNELS,
    K_SEGMENTS,
    N_MARKETING,
    N_POLICIES,
    TARGET_LOSS_RATIO,
)
from src.generate_data import generate_insurance, generate_marketing
from src.make_readme import render_readme

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
ASSETS = os.path.join(ROOT, "docs", "assets")


def fmt_pct(x, digits=1):
    return f"{x:.{digits}f}%"


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)

    # ---------------- Part A: marketing ----------------
    print("Generating 45,000 synthetic marketing records...")
    mkt = generate_marketing()
    assert len(mkt) == N_MARKETING

    print("Training response + uplift models...")
    resp = mm.train_response_model(mkt)
    uplift_df, _ = mm.train_uplift_two_model(mkt)
    lift = mm.decile_lift(uplift_df)
    seg = mm.segment_customers(mkt)

    print("Optimizing $500K budget across 4 channels (PuLP)...")
    econ = bo.channel_economics(uplift_df)
    opt = bo.optimize(econ)
    assert opt["lp"]["status"] == "Optimal"

    # ---------------- Part B: insurance ----------------
    print("Generating 595,000 synthetic auto policies...")
    ins = generate_insurance()
    assert len(ins) == N_POLICIES

    print("Training gradient-boosted claim model...")
    cm = im.train_claim_model(ins)
    tiers = im.risk_tiers(cm["test_frame"])
    tier_cutoffs = sorted(cm["test_frame"]["claim_score"].quantile([0.2, 0.4, 0.6, 0.8]).tolist())

    # ---------------- Figures ----------------
    print("Rendering figures...")
    fig.fig_decile_lift(
        lift["decile_response_rates"], lift["overall_response_rate"],
        os.path.join(ASSETS, "decile_lift.png"),
    )
    fig.fig_budget_allocation(
        opt, econ, os.path.join(ASSETS, "budget_allocation.png")
    )
    fig.fig_loss_ratio(
        tiers, TARGET_LOSS_RATIO, os.path.join(ASSETS, "loss_ratio_by_tier.png")
    )
    fig.fig_rate_adjustment(tiers, os.path.join(ASSETS, "rate_adjustment.png"))
    fig.fig_score_distribution(
        cm["test_frame"], os.path.join(ASSETS, "claim_score_distribution.png")
    )

    # ---------------- Metrics ----------------
    metrics = {
        "marketing": {
            "n_records": int(len(mkt)),
            "response_rate": float(mkt["response"].mean()),
            "response_rate_str": f"{mkt['response'].mean():.1%}",
            "response_model_auc": round(resp["auc"], 4),
            "response_model_auc_str": f"{resp['auc']:.2f}",
            "top3_lift_pct": round(lift["top3_lift_pct"], 2),
            "top3_lift_pct_str": fmt_pct(lift["top3_lift_pct"]),
            "top3_response_rate_str": f"{lift['top3_response_rate']:.1%}",
            "overall_response_rate_str": f"{lift['overall_response_rate']:.1%}",
            "n_segments": K_SEGMENTS,
            "cluster_shares": [round(s, 3) for s in seg["cluster_shares"]],
            "decile_response_rates": [round(r, 4) for r in lift["decile_response_rates"]],
        },
        "budget": {
            "total": BUDGET_TOTAL,
            "total_str": f"${BUDGET_TOTAL:,}",
            "n_channels": len(CHANNELS),
            "allocation": {ch: round(v) for ch, v in opt["lp"]["allocation"].items()},
            "allocation_str": {
                ch: f"${v:,.0f}" for ch, v in opt["lp"]["allocation"].items()
            },
            "lp_conversions": round(opt["lp"]["projected_conversions"]),
            "lp_conversions_str": f"{opt['lp']['projected_conversions']:,.0f}",
            "baseline_conversions": round(opt["baseline"]["projected_conversions"]),
            "baseline_conversions_str": f"{opt['baseline']['projected_conversions']:,.0f}",
            "conversion_gain_pct": round(opt["conversion_gain_pct"], 2),
            "conversion_gain_pct_str": fmt_pct(opt["conversion_gain_pct"]),
            "channel_economics": {
                ch: {
                    "cost_per_contact": float(econ.loc[ch, "cost_per_contact"]),
                    "marketable_universe": int(econ.loc[ch, "marketable_universe"]),
                    "expected_conv_rate": round(float(econ.loc[ch, "expected_conv_rate"]), 4),
                }
                for ch in CHANNELS
            },
        },
        "insurance": {
            "n_policies": int(len(ins)),
            "claim_rate": float(ins["claim"].mean()),
            "claim_rate_str": f"{ins['claim'].mean():.2%}",
            "auc": round(cm["auc"], 4),
            "auc_str": f"{cm['auc']:.2f}",
            "gini": round(cm["gini"], 4),
            "gini_str": f"{cm['gini']:.2f}",
            "target_loss_ratio": TARGET_LOSS_RATIO,
            "tier_score_cutoffs": [round(c, 4) for c in tier_cutoffs],
            "tiers": [
                {
                    "tier": int(i),
                    "policies": int(row["policies"]),
                    "loss_ratio": round(float(row["loss_ratio"]), 4),
                    "loss_ratio_str": f"{row['loss_ratio']:.1%}",
                    "observed_claim_rate_str": f"{row['observed_claim_rate']:.2%}",
                    "recommended_rate_change": round(float(row["recommended_rate_change"]), 4),
                    "recommended_rate_change_str": f"{row['recommended_rate_change']:+.1%}",
                    "action": str(row["action"]),
                }
                for i, row in tiers.iterrows()
            ],
        },
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------------- README ----------------
    readme = render_readme(metrics)
    with open(os.path.join(ROOT, "README.md"), "w") as f:
        f.write(readme)

    print("\nDone. Key measured results:")
    print(f"  top-3-decile lift : {metrics['marketing']['top3_lift_pct_str']}")
    print(f"  LP conversion gain: {metrics['budget']['conversion_gain_pct_str']}")
    print(f"  normalized Gini   : {metrics['insurance']['gini_str']}")
    print("  -> outputs/metrics.json, docs/assets/*.png, README.md")


if __name__ == "__main__":
    main()
