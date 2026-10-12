"""Part A - $500K budget allocation across 4 channels via linear programming.

Expected conversions per channel combine the predictive response scores with
per-channel contact costs; PuLP allocates the budget to maximize projected
conversions subject to per-channel capacity (reachable audience) constraints.
The baseline is an even $125K-per-channel split.
"""

import numpy as np
import pandas as pd
import pulp

from .config import (
    BUDGET_TOTAL,
    CHANNELS,
    CHANNEL_COST,
    CHANNEL_CONSENT_COL,
    CHANNEL_UNIVERSE,
)


def channel_economics(uplift_df: pd.DataFrame) -> pd.DataFrame:
    """Per-channel expected conversion rate and spend capacity.

    Expected conversion rates are estimated from the predictive model on the
    consenting customers in the 45K modeled records. The marketable universe
    (capacity) per channel is a media-planning input: the 45K records are the
    modeled sample, not the full contactable base.
    """
    rows = []
    for ch in CHANNELS:
        consent = CHANNEL_CONSENT_COL[ch]
        reachable = uplift_df[uplift_df[consent] == 1]
        conv_rate = float(reachable["pred_response"].mean())
        universe = int(CHANNEL_UNIVERSE[ch])
        rows.append(
            {
                "channel": ch,
                "cost_per_contact": CHANNEL_COST[ch],
                "marketable_universe": universe,
                "expected_conv_rate": conv_rate,
                "conv_per_dollar": conv_rate / CHANNEL_COST[ch],
                "max_spend": universe * CHANNEL_COST[ch],
            }
        )
    return pd.DataFrame(rows).set_index("channel")


def allocate_budget(econ: pd.DataFrame, budget: float = BUDGET_TOTAL) -> dict:
    """Maximize projected conversions with PuLP (fractional knapsack LP)."""
    prob = pulp.LpProblem("marketing_budget", pulp.LpMaximize)
    spend = {
        ch: prob.add_variable(
            f"spend_{ch}", lowBound=0, upBound=float(econ.loc[ch, "max_spend"])
        )
        for ch in CHANNELS
    }
    prob += pulp.lpSum(
        spend[ch] * float(econ.loc[ch, "conv_per_dollar"]) for ch in CHANNELS
    )
    prob += pulp.lpSum(spend[ch] for ch in CHANNELS) == budget
    stats = prob.solve(pulp.CPSAT(msg=False))
    status = "Optimal" if stats.has_solution else "Not solved"
    allocation = {ch: float(spend[ch].value()) for ch in CHANNELS}
    conversions = sum(
        allocation[ch] * float(econ.loc[ch, "conv_per_dollar"]) for ch in CHANNELS
    )
    return {
        "status": status,
        "allocation": allocation,
        "total_spend": float(sum(allocation.values())),
        "projected_conversions": float(conversions),
    }


def even_split_baseline(econ: pd.DataFrame, budget: float = BUDGET_TOTAL) -> dict:
    """Even $125K-per-channel split, capped at each channel's reachable capacity."""
    per_channel = budget / len(CHANNELS)
    allocation, conversions, spent = {}, 0.0, 0.0
    for ch in CHANNELS:
        s = min(per_channel, float(econ.loc[ch, "max_spend"]))
        allocation[ch] = s
        conversions += s * float(econ.loc[ch, "conv_per_dollar"])
        spent += s
    return {
        "allocation": allocation,
        "total_spend": float(spent),
        "projected_conversions": float(conversions),
    }


def optimize(econ: pd.DataFrame, budget: float = BUDGET_TOTAL) -> dict:
    """Run LP allocation and compare against the even-split baseline."""
    lp = allocate_budget(econ, budget)
    base = even_split_baseline(econ, budget)
    gain_pct = (lp["projected_conversions"] / base["projected_conversions"] - 1.0) * 100.0
    return {
        "lp": lp,
        "baseline": base,
        "conversion_gain_pct": float(gain_pct),
        "budget": float(budget),
        "n_channels": len(CHANNELS),
    }
