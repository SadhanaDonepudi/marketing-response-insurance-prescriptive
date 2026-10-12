"""Pipeline tests.

Covers: exact record counts, segmentation sanity, LP feasibility, model metric
sanity, and README-number consistency (every formatted number in README.md must
appear verbatim in outputs/metrics.json).

NOTE: run `python run_all.py` after any code change so outputs/metrics.json and
README.md reflect the current code before running pytest.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import budget_optimization as bo
from src import insurance_models as im
from src import marketing_models as mm
from src.config import BUDGET_TOTAL, CHANNELS, K_SEGMENTS, N_MARKETING, N_POLICIES
from src.generate_data import generate_insurance, generate_marketing


@pytest.fixture(scope="module")
def metrics():
    with open(os.path.join(ROOT, "outputs", "metrics.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def readme():
    with open(os.path.join(ROOT, "README.md")) as f:
        return f.read()


# ---------------- exact record counts ----------------

def test_marketing_record_count():
    df = generate_marketing()
    assert len(df) == 45_000 == N_MARKETING
    assert (df["is_synthetic"] == True).all()  # noqa: E712


def test_insurance_record_count():
    df = generate_insurance()
    assert len(df) == 595_000 == N_POLICIES
    assert (df["is_synthetic"] == True).all()  # noqa: E712


def test_metrics_counts_match(metrics):
    assert metrics["marketing"]["n_records"] == N_MARKETING
    assert metrics["insurance"]["n_policies"] == N_POLICIES


# ---------------- segmentation sanity ----------------

def test_segmentation_sanity():
    df = generate_marketing()
    seg = mm.segment_customers(df)
    assert len(seg["cluster_shares"]) == K_SEGMENTS == 5
    assert set(sorted(set(seg["labels"]))) == set(range(5))
    # no degenerate clusters
    assert min(seg["cluster_shares"]) > 0.01
    # shares sum to 1
    assert abs(sum(seg["cluster_shares"]) - 1.0) < 1e-9
    # profile covers every segment
    assert len(seg["profile"]) == 5


# ---------------- LP feasibility ----------------

@pytest.fixture(scope="module")
def optimization():
    df = generate_marketing()
    uplift_df, _ = mm.train_uplift_two_model(df)
    econ = bo.channel_economics(uplift_df)
    return bo.optimize(econ), econ


def test_lp_budget_sums_to_500k(optimization):
    opt, _ = optimization
    assert opt["lp"]["status"] == "Optimal"
    assert abs(opt["lp"]["total_spend"] - BUDGET_TOTAL) < 1.0
    assert abs(sum(opt["lp"]["allocation"].values()) - 500_000) < 1.0


def test_lp_four_channels(optimization):
    opt, econ = optimization
    assert opt["n_channels"] == 4
    assert set(opt["lp"]["allocation"].keys()) == set(CHANNELS)
    for ch in CHANNELS:
        alloc = opt["lp"]["allocation"][ch]
        assert alloc >= 0
        assert alloc <= float(econ.loc[ch, "max_spend"]) + 1e-6


def test_lp_beats_even_split(optimization):
    opt, _ = optimization
    assert opt["lp"]["projected_conversions"] > opt["baseline"]["projected_conversions"]
    assert opt["conversion_gain_pct"] > 0


# ---------------- model metric sanity ----------------

def test_response_model_sanity(metrics):
    auc = metrics["marketing"]["response_model_auc"]
    assert 0.5 < auc < 1.0
    lift = metrics["marketing"]["top3_lift_pct"]
    assert lift > 0  # targeting beats random


def test_decile_lift_monotone(metrics):
    rates = metrics["marketing"]["decile_response_rates"]
    assert len(rates) == 10
    # top deciles should respond at higher rates than bottom deciles on average
    assert sum(rates[:3]) / 3 > sum(rates[-3:]) / 3


def test_claim_model_sanity(metrics):
    auc = metrics["insurance"]["auc"]
    gini = metrics["insurance"]["gini"]
    assert 0.5 < auc < 1.0
    assert 0.0 < gini < 1.0
    assert abs(gini - (2 * auc - 1)) < 1e-9  # normalized Gini = 2*AUC - 1


def test_risk_tiers_sane(metrics):
    tiers = metrics["insurance"]["tiers"]
    assert len(tiers) == 5
    lrs = [t["loss_ratio"] for t in tiers]
    assert lrs == sorted(lrs), "loss ratios should rise with risk tier"
    for t in tiers:
        assert -0.20 - 1e-9 <= t["recommended_rate_change"] <= 0.30 + 1e-9
        assert t["action"] in ("increase rates", "decrease rates", "hold")


# ---------------- README-number consistency ----------------

def _formatted_strings(metrics):
    out = [
        metrics["marketing"]["top3_lift_pct_str"],
        metrics["marketing"]["top3_response_rate_str"],
        metrics["marketing"]["overall_response_rate_str"],
        metrics["marketing"]["response_rate_str"],
        metrics["marketing"]["response_model_auc_str"],
        metrics["budget"]["total_str"],
        metrics["budget"]["conversion_gain_pct_str"],
        metrics["budget"]["lp_conversions_str"],
        metrics["budget"]["baseline_conversions_str"],
        metrics["insurance"]["gini_str"],
        metrics["insurance"]["auc_str"],
        metrics["insurance"]["claim_rate_str"],
    ]
    out.extend(metrics["budget"]["allocation_str"].values())
    for t in metrics["insurance"]["tiers"]:
        out.extend([
            t["loss_ratio_str"],
            t["observed_claim_rate_str"],
            t["recommended_rate_change_str"],
        ])
    return out


def test_readme_numbers_match_metrics(metrics, readme):
    missing = [s for s in _formatted_strings(metrics) if s not in readme]
    assert not missing, f"README is missing measured values: {missing}"


def test_readme_synthetic_disclosure(readme):
    assert "synthetic" in readme.lower()
