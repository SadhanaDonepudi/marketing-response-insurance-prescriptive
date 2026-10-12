"""Synthetic data generators. All data is synthetic and clearly labeled as such.

- generate_marketing(): 45,000 bank marketing records (Part A)
- generate_insurance(): 595,000 auto insurance policies (Part B)

Both generators are deterministic given their seeds.
"""

import numpy as np
import pandas as pd

from .config import SEED_MARKETING, SEED_INSURANCE, N_MARKETING, N_POLICIES


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_marketing(n: int = N_MARKETING, seed: int = SEED_MARKETING) -> pd.DataFrame:
    """Generate synthetic bank marketing campaign records."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(46, 14, n), 18, 80).round(1)
    income_k = np.clip(rng.lognormal(np.log(72), 0.55, n), 18, 400).round(1)
    tenure_months = np.clip(rng.exponential(48, n), 1, 240).astype(np.int32)
    avg_balance = np.clip(rng.lognormal(np.log(8500), 0.9, n), 100, 250_000).round(0)
    num_products = np.clip(rng.poisson(1.8, n) + 1, 1, 5).astype(np.int8)
    prior_response = rng.binomial(1, 0.16, n).astype(np.int8)
    region = rng.choice(
        ["Northeast", "Midwest", "South", "West"], n, p=[0.22, 0.24, 0.30, 0.24]
    )
    email_consent = rng.binomial(1, 0.84, n).astype(np.int8)
    sms_consent = rng.binomial(1, 0.66, n).astype(np.int8)
    # Phone consent skews toward tenured, previously-responsive customers
    # (phone is the high-touch, high-conversion channel).
    p_phone = _expit(-1.6 + 0.006 * tenure_months + 0.6 * prior_response)
    phone_consent = rng.binomial(1, p_phone).astype(np.int8)
    mail_reachable = rng.binomial(1, 0.90, n).astype(np.int8)
    # Randomized campaign exposure (treatment) for uplift modeling.
    treatment = rng.binomial(1, 0.60, n).astype(np.int8)

    region_effect = np.select(
        [region == "Northeast", region == "Midwest", region == "South"],
        [0.05, 0.01, -0.03],
        default=0.0,
    )
    z = (
        -2.02
        - 0.008 * (age - 46)
        + 0.15 * np.log(income_k / 72)
        + 0.0018 * tenure_months
        + 0.12 * np.log(avg_balance / 8500)
        + 0.10 * (num_products - 2.4)
        + 0.40 * prior_response
        + region_effect
    )
    # Heterogeneous treatment effect -> what the uplift model must recover.
    tau = 0.30 + 0.25 * prior_response - 0.001 * tenure_months + 0.06 * (num_products >= 3)
    tau = np.clip(tau, 0.05, 1.00)
    # Unobserved heterogeneity caps how well any model can rank customers.
    z = z + rng.normal(0, 1.90, n)

    p = _expit(z + tau * treatment)
    response = rng.binomial(1, p).astype(np.int8)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1, dtype=np.int32),
            "age": age.astype(np.float32),
            "income_k": income_k.astype(np.float32),
            "tenure_months": tenure_months,
            "avg_balance": avg_balance.astype(np.float32),
            "num_products": num_products,
            "prior_response": prior_response,
            "region": pd.Categorical(region),
            "email_consent": email_consent,
            "sms_consent": sms_consent,
            "phone_consent": phone_consent,
            "mail_reachable": mail_reachable,
            "treatment": treatment,
            "response": response,
            "is_synthetic": True,
        }
    )
    assert len(df) == n
    return df


def generate_insurance(n: int = N_POLICIES, seed: int = SEED_INSURANCE) -> pd.DataFrame:
    """Generate synthetic auto insurance policies with claim outcomes."""
    rng = np.random.default_rng(seed)

    # Driver age: young / mid / senior mixture.
    age_group = rng.choice(["young", "mid", "senior"], n, p=[0.18, 0.55, 0.27])
    driver_age = np.where(
        age_group == "young",
        rng.integers(16, 26, n),
        np.where(
            age_group == "mid",
            np.clip(rng.normal(42, 9, n), 26, 60),
            np.clip(rng.normal(68, 6, n), 61, 88),
        ),
    ).astype(np.float32)

    vehicle_age = np.clip(rng.exponential(6, n), 0, 25).round(1).astype(np.float32)
    vehicle_type = rng.choice(
        ["sedan", "suv", "truck", "sports"], n, p=[0.42, 0.33, 0.15, 0.10]
    )
    territory = rng.choice(["urban", "suburban", "rural"], n, p=[0.45, 0.35, 0.20])
    annual_miles = np.clip(rng.lognormal(np.log(12000), 0.40, n), 1500, 45000).round(0)
    prior_claims = np.clip(rng.poisson(0.12, n), 0, 4).astype(np.int8)
    credit_band = rng.choice(
        ["excellent", "good", "fair", "poor"], n, p=[0.30, 0.38, 0.22, 0.10]
    )
    coverage_tier = rng.choice(["basic", "standard", "premium"], n, p=[0.35, 0.45, 0.20])

    terr_mult = np.select(
        [territory == "urban", territory == "rural"], [1.18, 0.92], default=1.0
    )
    veh_mult = np.select(
        [vehicle_type == "sports", vehicle_type == "truck", vehicle_type == "suv"],
        [1.32, 1.10, 1.05],
        default=1.0,
    )
    cov_mult = np.select(
        [coverage_tier == "premium", coverage_tier == "standard"], [1.35, 1.15], default=1.0
    )
    premium = (
        1450.0 * terr_mult * veh_mult * cov_mult * rng.lognormal(0, 0.08, n)
    ).round(2).astype(np.float32)

    age_effect = np.select(
        [age_group == "young", age_group == "senior"], [0.60, 0.20], default=0.0
    )
    veh_effect = np.select(
        [vehicle_type == "sports", vehicle_type == "truck", vehicle_type == "suv"],
        [0.44, 0.12, 0.042],
        default=0.0,
    )
    terr_effect = np.select(
        [territory == "urban", territory == "rural"], [0.28, -0.08], default=0.0
    )
    credit_effect = np.select(
        [credit_band == "poor", credit_band == "fair", credit_band == "excellent"],
        [0.32, 0.12, -0.08],
        default=0.0,
    )
    z = (
        -3.60
        + age_effect
        + 0.048 * vehicle_age
        + veh_effect
        + terr_effect
        + 0.24 * np.log(annual_miles / 12000)
        + 0.68 * prior_claims
        + credit_effect
    )
    p_claim = _expit(z)
    claim = rng.binomial(1, p_claim).astype(np.int8)
    severity = np.where(
        claim == 1, rng.lognormal(np.log(11500), 0.9, n).round(2), 0.0
    ).astype(np.float32)

    df = pd.DataFrame(
        {
            "policy_id": np.arange(1, n + 1, dtype=np.int32),
            "driver_age": driver_age,
            "vehicle_age": vehicle_age,
            "vehicle_type": pd.Categorical(vehicle_type),
            "territory": pd.Categorical(territory),
            "annual_miles": annual_miles.astype(np.float32),
            "prior_claims": prior_claims,
            "credit_band": pd.Categorical(credit_band),
            "coverage_tier": pd.Categorical(coverage_tier),
            "premium": premium,
            "claim": claim,
            "severity": severity,
            "is_synthetic": True,
        }
    )
    assert len(df) == n
    return df
