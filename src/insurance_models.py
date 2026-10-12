"""Part B - auto insurance claim-likelihood modeling and pricing recommendations."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import SEED_SPLIT, N_RISK_TIERS, TARGET_LOSS_RATIO

FEATURES = [
    "driver_age",
    "vehicle_age",
    "vehicle_type",
    "territory",
    "annual_miles",
    "prior_claims",
    "credit_band",
    "coverage_tier",
]
CAT_FEATURES = ["vehicle_type", "territory", "credit_band", "coverage_tier"]


def normalized_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Normalized Gini = 2 * AUC - 1 (Somers' D for binary outcomes)."""
    return float(2.0 * roc_auc_score(y_true, y_score) - 1.0)


def train_claim_model(df: pd.DataFrame, seed: int = SEED_SPLIT):
    """Gradient-boosted claim-likelihood model (HistGradientBoosting)."""
    X = df[FEATURES].copy()
    y = df["claim"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        categorical_features=CAT_FEATURES,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    scores = model.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, scores))
    gini = normalized_gini(y_te, scores)

    te = df.loc[X_te.index].copy()
    te["claim_score"] = scores
    return {
        "model": model,
        "auc": auc,
        "gini": gini,
        "X_test": X_te,
        "y_test": y_te,
        "test_frame": te,
    }


def risk_tiers(test_frame: pd.DataFrame, n_tiers: int = N_RISK_TIERS) -> pd.DataFrame:
    """Segment the scored test population into risk tiers and price them.

    Loss ratio = incurred claim severity / earned premium per tier.
    Recommended rate adjustment moves each tier toward the target loss ratio:
        adj = tier_lr / target_lr - 1, clipped to [-20%, +30%].
    High-loss-ratio tiers get rate increases; profitable low-risk tiers get
    decreases to stay competitive.
    """
    tf = test_frame.copy()
    tf["risk_tier"] = pd.qcut(
        tf["claim_score"].rank(method="first"), n_tiers, labels=False
    ) + 1
    agg = (
        tf.groupby("risk_tier")
        .agg(
            policies=("policy_id", "size"),
            earned_premium=("premium", "sum"),
            incurred_loss=("severity", "sum"),
            claim_count=("claim", "sum"),
            avg_score=("claim_score", "mean"),
        )
        .round(2)
    )
    agg["loss_ratio"] = (agg["incurred_loss"] / agg["earned_premium"]).round(4)
    agg["observed_claim_rate"] = (agg["claim_count"] / agg["policies"]).round(4)
    adj = agg["loss_ratio"] / TARGET_LOSS_RATIO - 1.0
    agg["recommended_rate_change"] = adj.clip(-0.20, 0.30).round(4)
    agg["action"] = np.select(
        [agg["recommended_rate_change"] >= 0.05, agg["recommended_rate_change"] <= -0.05],
        ["increase rates", "decrease rates"],
        default="hold",
    )
    return agg
