"""Rendered matplotlib figures for the README and the Power BI report spec."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def fig_decile_lift(decile_rates, overall_rate, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = np.arange(1, 11)
    colors = ["#c0392b" if i <= 3 else "#7f8c8d" for i in xs]
    ax.bar(xs, decile_rates, color=colors, edgecolor="white")
    ax.axhline(overall_rate, color="#2c3e50", linestyle="--", linewidth=1.5,
               label=f"Overall response rate ({overall_rate:.1%})")
    ax.set_xlabel("Uplift decile (1 = highest predicted uplift)")
    ax.set_ylabel("Treated response rate")
    ax.set_title("Response rate by uplift decile — top 3 deciles highlighted")
    ax.set_xticks(xs)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_budget_allocation(opt_result, econ, path):
    labels = [c.replace("_", " ").title() for c in econ.index]
    x = np.arange(len(labels))
    w = 0.35
    lp_vals = [opt_result["lp"]["allocation"][c] / 1000 for c in econ.index]
    base_vals = [opt_result["baseline"]["allocation"][c] / 1000 for c in econ.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w / 2, base_vals, w, label="Even split ($125K/channel)", color="#7f8c8d")
    ax.bar(x + w / 2, lp_vals, w, label="LP-optimized", color="#2980b9")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Budget ($ thousands)")
    ax.set_title("Marketing budget allocation: even split vs LP optimization ($500K)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_loss_ratio(tiers: pd.DataFrame, target_lr: float, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    tiers_sorted = tiers.sort_index()
    colors = ["#27ae60" if lr <= target_lr else "#c0392b"
              for lr in tiers_sorted["loss_ratio"]]
    ax.bar([f"Tier {i}" for i in tiers_sorted.index], tiers_sorted["loss_ratio"],
           color=colors, edgecolor="white")
    ax.axhline(target_lr, color="#2c3e50", linestyle="--", linewidth=1.5,
               label=f"Target loss ratio ({target_lr:.0%})")
    ax.set_ylabel("Loss ratio (incurred / earned premium)")
    ax.set_title("Loss ratio by predicted-claim risk tier (test set)")
    ax.legend(frameon=False)
    for i, v in enumerate(tiers_sorted["loss_ratio"]):
        ax.text(i, v + 0.012, f"{v:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_rate_adjustment(tiers: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    tiers_sorted = tiers.sort_index()
    vals = tiers_sorted["recommended_rate_change"] * 100
    colors = ["#c0392b" if v > 0 else "#27ae60" for v in vals]
    ax.bar([f"Tier {i}" for i in tiers_sorted.index], vals, color=colors, edgecolor="white")
    ax.axhline(0, color="#2c3e50", linewidth=1)
    ax.set_ylabel("Recommended rate change (%)")
    ax.set_title("Recommended rate adjustment by risk tier")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.6 if v >= 0 else -1.4), f"{v:+.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_score_distribution(test_frame: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(test_frame.loc[test_frame["claim"] == 0, "claim_score"], bins=60,
            alpha=0.6, label="No claim", color="#7f8c8d", density=True)
    ax.hist(test_frame.loc[test_frame["claim"] == 1, "claim_score"], bins=60,
            alpha=0.7, label="Claim", color="#c0392b", density=True)
    ax.set_xlabel("Predicted claim probability")
    ax.set_ylabel("Density")
    ax.set_title("Claim score distribution by actual outcome (test set)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
