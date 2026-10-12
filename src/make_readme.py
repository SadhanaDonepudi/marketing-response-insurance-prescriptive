"""Render README.md from outputs/metrics.json.

Every number in the README is filled from the measured pipeline outputs, so
README numbers match the code's actual results by construction.
"""

README_TEMPLATE = """# Marketing Response & Insurance Claims Prescriptive Analytics

**All data in this repo is synthetic** (generated with fixed random seeds for
reproducibility). Two-part project: (A) bank marketing response & uplift modeling
with LP budget optimization, and (B) auto insurance claim-likelihood modeling
translated into pricing recommendations.

`Python (scikit-learn, PuLP)` · `R` · `SQL` · `Power BI` (report spec)

## Measured results (from `python run_all.py`)

| Claim | Measured |
|---|---|
| Marketing records modeled | **{mkt_n:,}** synthetic bank marketing records |
| Response lift, top 3 uplift deciles | **{top3_lift}** (treated response rate {top3_rate} vs overall {overall_rate}) |
| Response model AUC (treated pop.) | **{resp_auc}** |
| Budget optimization | **{budget}** across **{n_channels}** channels via linear programming (PuLP) |
| Projected conversions vs even split | **{gain} more conversions** ({lp_conv} vs {base_conv} baseline) |
| Auto policies scored | **{ins_n:,}** synthetic auto policies |
| Claim-likelihood model, normalized Gini | **{gini}** (AUC {ins_auc}, gradient boosting) |
| Claim rate in synthetic book | **{claim_rate}** |

## Part A — Marketing response, uplift & budget optimization

1. **Synthetic data** (`src/generate_data.py`): {mkt_n:,} bank marketing records —
   demographics, channel exposure/consent, tenure, balances, prior response, and a
   randomized campaign exposure flag for uplift modeling.
2. **Response model**: gradient boosting (`HistGradientBoostingClassifier`) trained
   on the treated population — AUC {resp_auc}.
3. **Uplift model** (two-model approach): P(response|treated) − P(response|control).
   Customers ranked by predicted uplift; the **top 3 deciles respond at a
   {top3_lift} higher rate** than average ({top3_rate} vs {overall_rate}).
   ![Decile lift](docs/assets/decile_lift.png)
4. **k-means segmentation** ({n_segments} segments) on standardized customer features
   for targeting; cluster shares: {cluster_shares}.
5. **Budget optimization** (`src/budget_optimization.py`): PuLP linear program
   allocates {budget} across email, SMS, phone, and direct mail to maximize
   projected conversions, combining predicted response scores with per-channel
   contact costs and marketable-universe capacity constraints. The LP allocation
   projects **{gain} more conversions** than an even ${even_each:,.0f}-per-channel split.
   ![Budget allocation](docs/assets/budget_allocation.png)

LP allocation: {alloc_str}

## Part B — Auto insurance claim likelihood & pricing

1. **Synthetic data**: {ins_n:,} auto policies — driver age, vehicle, territory,
   annual mileage, prior claims, credit band, coverage tier, premium, claim
   indicator, and severity.
2. **Claim-likelihood model**: gradient boosting — **normalized Gini {gini}**
   (Gini = 2·AUC − 1; AUC {ins_auc}) on a held-out test set.
   ![Score distribution](docs/assets/claim_score_distribution.png)
3. **Risk tiers → pricing**: scored policies are grouped into 5 risk tiers
   (quintiles of predicted claim probability). Loss ratio = incurred / earned
   premium per tier; recommended rate change = tier LR / {target_lr:.0%} target − 1
   (clipped to −20% / +30%).
   ![Loss ratio by tier](docs/assets/loss_ratio_by_tier.png)
   ![Rate adjustment](docs/assets/rate_adjustment.png)

| Risk tier | Policies | Loss ratio | Observed claim rate | Recommended rate change | Action |
|---|---|---|---|---|---|
{tiers_table}

4. **Power BI report spec** (`docs/powerbi_report_spec.md`): full report design —
   pages, DAX measures, visuals (the charts above are the rendered outputs),
   and refresh/RLS notes.

## Repo layout

```
├── run_all.py                # end-to-end pipeline (data → models → figures → README)
├── src/
│   ├── generate_data.py      # synthetic data generators (45K marketing / 595K policies)
│   ├── marketing_models.py   # response model, two-model uplift, decile lift, k-means
│   ├── budget_optimization.py# PuLP LP: $500K across 4 channels vs even-split baseline
│   ├── insurance_models.py   # gradient-boosted claim model, normalized Gini, risk tiers
│   ├── figures.py            # matplotlib renderings for README + Power BI spec
│   └── make_readme.py        # README template filled from measured metrics
├── sql/analysis_queries.sql  # companion SQL analyses (decile lift, loss ratios)
├── r/uplift_validation.R     # supplementary R validation script
├── docs/
│   ├── data_dictionary.md    # data dictionary for both datasets
│   ├── powerbi_report_spec.md# Power BI report design spec
│   └── assets/               # rendered figures
├── outputs/metrics.json      # every measured number (single source of truth)
└── tests/test_pipeline.py    # pytest suite
```

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py        # regenerates outputs/metrics.json, docs/assets/*.png, README.md
pytest                   # runs the test suite
```

> **Honesty note:** all datasets are synthetic and labeled as such
> (`is_synthetic = True` on every record). Every number in this README is the
> measured output of the code in this repo — see `outputs/metrics.json`.
> Resume bullets describe the same project; where this repo's reproducible
> numbers differ, the repo reports the measured value.
"""


def render_readme(m: dict) -> str:
    tiers_rows = "\n".join(
        f"| Tier {t['tier']} | {t['policies']:,} | {t['loss_ratio_str']} | "
        f"{t['observed_claim_rate_str']} | {t['recommended_rate_change_str']} | {t['action']} |"
        for t in m["insurance"]["tiers"]
    )
    alloc_str = ", ".join(
        f"{ch.replace('_', ' ')}: {m['budget']['allocation_str'][ch]}"
        for ch in ("email", "sms", "phone", "direct_mail")
    )
    return README_TEMPLATE.format(
        mkt_n=m["marketing"]["n_records"],
        top3_lift=m["marketing"]["top3_lift_pct_str"],
        top3_rate=m["marketing"]["top3_response_rate_str"],
        overall_rate=m["marketing"]["overall_response_rate_str"],
        resp_auc=m["marketing"]["response_model_auc_str"],
        budget=m["budget"]["total_str"],
        n_channels=m["budget"]["n_channels"],
        gain=m["budget"]["conversion_gain_pct_str"],
        lp_conv=m["budget"]["lp_conversions_str"],
        base_conv=m["budget"]["baseline_conversions_str"],
        ins_n=m["insurance"]["n_policies"],
        gini=m["insurance"]["gini_str"],
        ins_auc=m["insurance"]["auc_str"],
        claim_rate=m["insurance"]["claim_rate_str"],
        n_segments=m["marketing"]["n_segments"],
        cluster_shares=", ".join(f"{s:.1%}" for s in m["marketing"]["cluster_shares"]),
        even_each=m["budget"]["total"] / m["budget"]["n_channels"],
        alloc_str=alloc_str,
        target_lr=m["insurance"]["target_loss_ratio"],
        tiers_table=tiers_rows,
    )
