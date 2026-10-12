-- Companion SQL analyses for the synthetic marketing + insurance datasets.
-- Table schemas: docs/data_dictionary.md
-- Run after loading the generated tables as marketing_customers / auto_policies.

-- ---------------------------------------------------------------------------
-- 1. Campaign response rate by uplift decile (validates the 31% top-3-decile lift)
--    Assumes a scored table column `uplift_score` (see src/marketing_models.py).
-- ---------------------------------------------------------------------------
WITH ranked AS (
    SELECT
        response,
        treatment,
        NTILE(10) OVER (ORDER BY uplift_score DESC) AS uplift_decile
    FROM marketing_customers
),
decile_rates AS (
    SELECT
        uplift_decile,
        AVG(CASE WHEN treatment = 1 THEN response END) AS treated_response_rate,
        COUNT(*) AS n
    FROM ranked
    GROUP BY uplift_decile
),
overall AS (
    SELECT AVG(response) AS overall_rate FROM marketing_customers
)
SELECT
    d.uplift_decile,
    d.treated_response_rate,
    o.overall_rate,
    d.treated_response_rate / o.overall_rate - 1 AS lift_vs_overall
FROM decile_rates d
CROSS JOIN overall o
ORDER BY d.uplift_decile;

-- ---------------------------------------------------------------------------
-- 2. Channel economics: reachable base and observed conversion rate per channel
--    (inputs to the PuLP budget optimization in src/budget_optimization.py)
-- ---------------------------------------------------------------------------
SELECT 'email'       AS channel, 1.25 AS cost_per_contact,
       AVG(email_consent)  AS reach_rate, AVG(response) AS conv_rate
FROM marketing_customers
UNION ALL
SELECT 'sms', 0.85, AVG(sms_consent), AVG(response) FROM marketing_customers
UNION ALL
SELECT 'phone', 7.50, AVG(phone_consent), AVG(response) FROM marketing_customers
UNION ALL
SELECT 'direct_mail', 4.25, AVG(mail_reachable), AVG(response) FROM marketing_customers;

-- ---------------------------------------------------------------------------
-- 3. Loss ratio and recommended rate change by claim-score risk tier
--    (mirrors src/insurance_models.py::risk_tiers; target loss ratio = 0.62)
-- ---------------------------------------------------------------------------
WITH scored AS (
    SELECT
        premium,
        severity,
        claim,
        NTILE(5) OVER (ORDER BY claim_score) AS risk_tier
    FROM auto_policies
),
tier AS (
    SELECT
        risk_tier,
        COUNT(*) AS policies,
        SUM(premium)  AS earned_premium,
        SUM(severity) AS incurred_loss,
        AVG(claim)    AS observed_claim_rate
    FROM scored
    GROUP BY risk_tier
)
SELECT
    risk_tier,
    policies,
    incurred_loss / earned_premium AS loss_ratio,
    observed_claim_rate,
    GREATEST(-0.20, LEAST(0.30, (incurred_loss / earned_premium) / 0.62 - 1))
        AS recommended_rate_change
FROM tier
ORDER BY risk_tier;

-- ---------------------------------------------------------------------------
-- 4. Data-quality checks on the synthetic extracts
-- ---------------------------------------------------------------------------
SELECT 'marketing' AS dataset,
       COUNT(*) AS n_rows,
       SUM(CASE WHEN is_synthetic IS NOT TRUE THEN 1 ELSE 0 END) AS non_synthetic_rows,
       COUNT(*) - COUNT(customer_id) AS null_ids
FROM marketing_customers
UNION ALL
SELECT 'insurance',
       COUNT(*),
       SUM(CASE WHEN is_synthetic IS NOT TRUE THEN 1 ELSE 0 END),
       COUNT(*) - COUNT(policy_id)
FROM auto_policies;
