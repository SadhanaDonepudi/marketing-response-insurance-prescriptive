# Power BI Report Spec — Auto Insurance Claim Risk & Pricing Recommendations

Design specification for the Part B Power BI report. The charts below are the
rendered matplotlib outputs from this repo's pipeline (`python run_all.py`);
in Power BI they are rebuilt as native visuals over the `auto_policies` table
(schema: `docs/data_dictionary.md`). No `.pbix` is checked in — this spec is
the build blueprint.

## 1. Report pages

### Page 1 — Executive Summary
- **KPI cards:** Policies scored, Earned premium, Incurred loss, Overall loss ratio,
  Model normalized Gini.
- **Bar chart:** Loss ratio by risk tier with a constant line at the 62% target.
  Rendered output: ![Loss ratio by tier](assets/loss_ratio_by_tier.png)
- **Bar chart:** Recommended rate adjustment by risk tier.
  Rendered output: ![Rate adjustment](assets/rate_adjustment.png)
- **Slicers:** Territory, Vehicle type, Coverage tier.

### Page 2 — Risk Segmentation
- **Histogram:** Predicted claim-probability distribution, split by actual claim
  outcome (model separation check).
  Rendered output: ![Score distribution](assets/claim_score_distribution.png)
- **Table:** Risk tier × policies, earned premium, incurred loss, loss ratio,
  observed claim rate, recommended rate change, action.
- **Scatter:** Average predicted score vs observed claim rate by tier
  (calibration check).

### Page 3 — Pricing Simulator
- **What-if parameters:** `Target loss ratio` (50%–75%), `Max increase` (0%–30%),
  `Max decrease` (0%–20%).
- **Table:** per-tier recommended rate change recomputed from the parameters.
- **Waterfall:** earned premium impact of applying the recommended adjustments.

### Page 4 — Model Performance & Monitoring
- **KPI cards:** AUC, normalized Gini (2·AUC − 1), KS statistic.
- **Line chart:** cumulative gains / Lorenz curve on the scored test set.
- **Drift placeholders:** PSI tiles per input feature for monthly refresh monitoring.

## 2. DAX measures

```dax
Earned Premium = SUM ( auto_policies[premium] )

Incurred Loss = SUM ( auto_policies[severity] )

Loss Ratio = DIVIDE ( [Incurred Loss], [Earned Premium] )

Claim Rate = DIVIDE ( COUNTROWS ( FILTER ( auto_policies, auto_policies[claim] = 1 ) ),
                      COUNTROWS ( auto_policies ) )

Risk Tier =
VAR s = SELECTEDVALUE ( auto_policies[claim_score] )
RETURN
    SWITCH ( TRUE (),
        s < 0.0372, 1,
        s < 0.0462, 2,
        s < 0.0573, 3,
        s < 0.0783, 4,
        5 )
-- cutoffs = claim-score quintile boundaries on the scored test set
-- (see outputs/metrics.json -> insurance.tier_score_cutoffs)

Recommended Rate Change =
VAR lr = [Loss Ratio]
VAR tgt = [Target Loss Ratio Value]      -- from what-if parameter
RETURN
    MAX ( -[Max Decrease Value],
        MIN ( [Max Increase Value], DIVIDE ( lr, tgt ) - 1 ) )

Pricing Action =
SWITCH ( TRUE (),
    [Recommended Rate Change] >= 0.05, "increase rates",
    [Recommended Rate Change] <= -0.05, "decrease rates",
    "hold" )
```

## 3. Data model & refresh

- Single fact table `auto_policies`; conformed dimensions for territory,
  vehicle type, credit band, coverage tier (or handle as model categories).
- `claim_score` and `risk_tier` are produced by the Python pipeline
  (`src/insurance_models.py`) and loaded as columns — scoring stays in the
  validated model artifact, not in DAX.
- Scheduled refresh: monthly, after the claim extract closes; append-only
  partitions by policy term.

## 4. Row-level security

- `rls_territory`: `[territory] = USERPRINCIPALNAME ()`-mapped territory for
  regional underwriting managers.
- Executive role: full access, no filter.

## 5. Business read (from the rendered outputs)

The model separates risk cleanly: loss ratios rise monotonically from 31.5%
in the lowest-risk tier to 93.6% in the highest-risk tier (target: 62%).
Tiers 1–3 earn the maximum −20% rate decrease and tier 4 a −5.4% decrease
(profitable, competitive pricing on good risks); the top-risk tier earns the
maximum +30% increase. The pricing simulator page lets underwriting re-run
these recommendations under different targets.
