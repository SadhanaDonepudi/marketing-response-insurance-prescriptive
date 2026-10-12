"""Shared configuration for the marketing + insurance prescriptive analytics repo."""

SEED_MARKETING = 42
SEED_INSURANCE = 123
SEED_SPLIT = 7

N_MARKETING = 45_000
N_POLICIES = 595_000

BUDGET_TOTAL = 500_000
CHANNELS = ["email", "sms", "phone", "direct_mail"]

# Cost to contact one customer through each channel (USD).
CHANNEL_COST = {
    "email": 1.25,
    "sms": 0.85,
    "phone": 7.50,
    "direct_mail": 4.25,
}

# Consent flag in the marketing table that makes a customer reachable per channel.
CHANNEL_CONSENT_COL = {
    "email": "email_consent",
    "sms": "sms_consent",
    "phone": "phone_consent",
    "direct_mail": "mail_reachable",
}

# Marketable universe (contacts) per channel - a media-planning input. The 45K
# modeled records are the estimation sample; the addressable base is larger.
# Capacities are set so the $500K budget is feasible but the LP must trade off
# across channels (cheap digital caps bind before the budget is exhausted).
CHANNEL_UNIVERSE = {
    "email": 148_800,
    "sms": 160_000,
    "phone": 20_000,
    "direct_mail": 40_000,
}

K_SEGMENTS = 5          # k-means customer segments
N_RISK_TIERS = 5        # insurance risk tiers (quintiles of predicted claim prob.)
TARGET_LOSS_RATIO = 0.62  # pricing target used for rate recommendations
