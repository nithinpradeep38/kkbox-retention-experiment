# KKBox Subscription Retention Experimentation Platform

An end-to-end product data science project — churn modeling, A/B test
design with economic justification, statistical analysis, and a
segment-targeted business recommendation — built on the
[KKBox Churn Prediction dataset](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data).

**Live demo:** [kkbox-retention.streamlit.app](https://kkbox-retention.streamlit.app) *(add once deployed)*

---

## Headline results

| Metric | Value |
|---|---|
| Eligible experiment population | 205,883 users |
| Renewal rate lift | **+11.99pp** (95% CI: [11.56pp, 12.42pp]) |
| p-value | < 0.0001 |
| Net revenue impact | **+1,254,280 NTD** after cannibalization |
| Decision | **GO** — targeted 10% discount, High-risk segment only |

---

## Architecture

![Architecture](assets/architecture.png)

---

## Project structure

```
kkbox-retention-experiment/
├── local_preprocessing/       # DuckDB scripts — trim 28GB locally before upload
│   ├── 00_local_trim.py
│   ├── explore_before_trim.py
│   ├── explore_windows.py
│   └── quick_overlap_check.py
├── notebooks/                 # Databricks notebooks (PySpark + scikit-learn)
│   ├── 01_data_engineering.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_churn_model.ipynb
│   ├── 05_experiment_design.ipynb
│   ├── 06_experiment_analysis.ipynb
│   └── 08_business_recommendation.ipynb
├── app/                       # Streamlit dashboard
│   ├── app.py
│   └── data/                  # Pre-computed results (no raw data committed)
├── assets/
│   └── architecture.png
├── requirements.txt
└── README.md
```

---

## Key findings

### 1. Churn is driven by subscription mechanics, not engagement

EDA across 970K labeled users found three strong churn signals:

| Signal | Churn rate | vs baseline (8.99%) |
|---|---|---|
| Auto-renew OFF | 29.48% | 3.3× |
| Ever cancelled | 27.21% | 3.0× |
| Zero-payment history | 35.27% | 6.0× |
| No visible transaction | 84.29% | 9.4× |

Engagement level (listening hours, active days) and tenure showed
**no meaningful relationship with churn** - flat across all buckets.
This was confirmed by logistic regression feature importance
(ROC-AUC 0.87): subscription mechanics dominate; engagement features
contribute negligible signal.

### 2. Discount rate requires economic justification

A lift × discount matrix identified 10% as the maximum economically
viable discount given an expected 10pp+ lift:

- Break-even discount at 10pp lift: 14.46%
- Chosen discount: **10%** (4.46pp buffer above break-even)

### 3. Experiment result: significant lift, net positive revenue

- Recovered lift: **+11.99pp** (injected truth: +12pp)
- 95% CI: **[11.56pp, 12.42pp]** entire CI above 10pp MDE
- Net revenue: **+1.25M NTD** (incremental revenue +2.77M NTD,
  cannibalization cost -1.52M NTD)
- Treatment RPU (160.18 NTD) > Control RPU (147.98 NTD)

All 14 pre-specified segment tests are statistically significant
and net positive. Lift is uniform across segments (~11.5–13.6pp),
consistent with the finding that the churn model has flat calibration
within the High-risk group.

---

## Targeting recommendation

Offer the 10% discount to users satisfying **all three**:
1. `risk_segment = High` (top tercile churn probability)
2. `membership_expire_date` within next 30 days
3. `has_visible_transaction = 1` (exclude already-churned users)

**Prioritize** within the eligible population:
- `ever_cancelled = 1` : 13.61pp lift (highest)
- `tenure < 6 months` :  12.70pp lift (habit formation window)

**Do not** offer to Low/Medium risk segments- cannibalization
economics are deeply negative at their ~96% baseline renewal rate.

---

## Tech stack

| Layer | Tools |
|---|---|
| Local preprocessing | Python, DuckDB |
| Compute | Databricks, PySpark, Spark SQL |
| Storage | Delta Lake, Unity Catalog |
| Feature engineering | PySpark window functions |
| Churn model | scikit-learn (logistic regression) |
| Experiment & stats | Python, SciPy, statsmodels |
| Dashboard | Streamlit |

---

## Running the Streamlit app locally

```bash
git clone https://github.com/nithinpradeep38/kkbox-retention-experiment
cd kkbox-retention-experiment
pip install -r requirements.txt
streamlit run app/app.py
```



## Data

Raw KKBox data is not committed (Kaggle license). Download from:
[kaggle.com/competitions/kkbox-churn-prediction-challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data)

The `app/data/` folder contains pre-computed summary CSVs only.

---

## Author

Nithin Pradeep · [github.com/nithinpradeep38](https://github.com/nithinpradeep38)