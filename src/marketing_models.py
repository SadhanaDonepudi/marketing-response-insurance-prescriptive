"""Part A - marketing response modeling, uplift modeling, and segmentation."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import SEED_SPLIT, K_SEGMENTS

FEATURES = [
    "age",
    "income_k",
    "tenure_months",
    "avg_balance",
    "num_products",
    "prior_response",
    "region",
]
CAT_FEATURES = ["region"]


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    return pd.get_dummies(X, columns=CAT_FEATURES, drop_first=True, dtype=np.float32)


def train_response_model(df: pd.DataFrame, seed: int = SEED_SPLIT):
    """Gradient-boosted response model on the treated (campaigned) population."""
    treated = df[df["treatment"] == 1].reset_index(drop=True)
    X = _encode(treated)
    y = treated["response"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_depth=4, random_state=seed
    )
    model.fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    return {
        "model": model,
        "auc": auc,
        "X_test": X_te,
        "y_test": y_te,
        "test_index": X_te.index,
        "treated_test": treated.loc[X_te.index],
    }


def train_uplift_two_model(df: pd.DataFrame, seed: int = SEED_SPLIT):
    """Two-model uplift: P(response|treated) - P(response|control)."""
    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]
    X_t, y_t = _encode(treated), treated["response"].to_numpy()
    X_c, y_c = _encode(control), control["response"].to_numpy()

    m_t = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_depth=4, random_state=seed
    ).fit(X_t, y_t)
    m_c = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_depth=4, random_state=seed + 1
    ).fit(X_c, y_c)

    X_all = _encode(df)
    uplift = m_t.predict_proba(X_all)[:, 1] - m_c.predict_proba(X_all)[:, 1]
    out = df[["treatment", "response", "email_consent", "sms_consent",
              "phone_consent", "mail_reachable"]].copy()
    out["uplift_score"] = uplift.astype(np.float32)
    out["pred_response"] = m_t.predict_proba(X_all)[:, 1].astype(np.float32)
    return out, {"model_treated": m_t, "model_control": m_c}


def decile_lift(uplift_df: pd.DataFrame) -> dict:
    """Response lift in the top 3 deciles vs the overall response rate.

    Customers are ranked by predicted uplift; lift = (response rate of treated
    customers in the top 3 deciles / overall response rate) - 1.
    """
    ranked = uplift_df.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    cutoff = int(len(ranked) * 0.30)
    top3 = ranked.iloc[:cutoff]
    top3_treated = top3[top3["treatment"] == 1]
    overall_rate = float(ranked["response"].mean())
    top3_rate = float(top3_treated["response"].mean())
    lift = (top3_rate / overall_rate - 1.0) * 100.0

    # Per-decile treated response rates (for the chart / report).
    ranked["decile"] = pd.qcut(
        ranked["uplift_score"].rank(method="first"), 10, labels=False
    )
    decile_rates = (
        ranked[ranked["treatment"] == 1].groupby("decile")["response"].mean()
    )
    decile_rates.index = 10 - decile_rates.index  # 1 = top decile
    decile_rates = decile_rates.sort_index()

    return {
        "overall_response_rate": overall_rate,
        "top3_response_rate": top3_rate,
        "top3_lift_pct": float(lift),
        "top3_n": int(len(top3)),
        "decile_response_rates": [float(v) for v in decile_rates.values],
    }


def segment_customers(df: pd.DataFrame, k: int = K_SEGMENTS, seed: int = SEED_SPLIT):
    """k-means segmentation on standardized numeric customer features."""
    num_cols = ["age", "income_k", "tenure_months", "avg_balance", "num_products"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[num_cols].to_numpy(dtype=np.float64))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(Xs)
    sizes = pd.Series(labels).value_counts(normalize=True).sort_index()
    profile = (
        df.assign(segment=labels)
        .groupby("segment")
        .agg(
            n=("customer_id", "size"),
            avg_age=("age", "mean"),
            avg_income_k=("income_k", "mean"),
            avg_tenure_months=("tenure_months", "mean"),
            avg_balance=("avg_balance", "mean"),
            response_rate=("response", "mean"),
        )
        .round(3)
    )
    return {
        "labels": labels,
        "kmeans": km,
        "cluster_shares": [float(s) for s in sizes.values],
        "profile": profile,
        "inertia": float(km.inertia_),
    }
