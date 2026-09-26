![Tests](https://github.com/sunnysavarna108-commits/churn-roi-app-v2/actions/workflows/tests.yml/badge.svg)

# 📉 Customer Churn & Retention ROI

Predicts a telecom customer's churn probability using a tuned XGBoost model, then translates that prediction into a business decision: is it worth spending money on a retention campaign for this customer, and at what threshold does targeting the whole customer base maximize profit?

**🔗 Live app:** [Open the app](https://churn-roi-app-v2-4skxsvfpetvweqdcdpcaah.streamlit.app/)

**📓 Training notebook:** [01_data_exploration.ipynb](./01_data_exploration.ipynb)

![App screenshot](screenshot.png)

---

## What it does

This isn't just a churn classifier — it's a decision-support tool. A model that says "this customer is 65% likely to churn" isn't useful on its own; this app answers the follow-up question: *should we spend money trying to keep them?*

### 🧍 Single customer
Enter one customer's details and get:
- Predicted churn probability and risk level
- Expected value saved, net gain, and ROI for a retention campaign
- The break-even campaign success rate needed to make the campaign worthwhile
- An interactive chart showing how net gain changes across every possible campaign success rate

### 📁 Batch scoring (CSV)
Upload a CSV of many customers and get:
- Every customer scored at once, with risk level and expected value saved
- Portfolio-level economics: total cost, total value saved, total ROI
- **Profit-maximizing threshold**: an interactive chart showing total portfolio profit at every possible targeting threshold, with the optimal point marked and compared to your current setting
- A downloadable CSV with all scores and recommendations added

### 📊 What drives churn
Global feature importance from the XGBoost model, grouped back into the original customer attributes (e.g. `Contract`, `InternetService`) rather than raw one-hot columns.

### 🔎 Why this score
- **Per-customer explanation** using the model's real TreeSHAP contribution values — shows exactly which attributes pushed this specific customer's risk up or down, and by how much
- **What-if comparison**: pick an attribute (e.g. Contract, Payment Method) and see the model's actual re-predicted probability for every possible value, holding everything else constant — a real prediction for each option, not an estimate

---

## The model

- **Data:** Telco customer churn dataset (19 customer attributes: demographics, services, account/billing details)
- **Pipeline:** `imblearn.Pipeline` — median/mode imputation → one-hot encoding → **SMOTE** (to handle class imbalance, since churners are a minority class) → **XGBoost classifier**
- **Tuning:** hyperparameters selected via search on the training set (`best_pipeline`)

### Performance (held-out test set, 1,409 customers, 26% churners)

| Metric | Value |
|---|---|
| ROC-AUC | **0.857** |
| Precision (churn class) | 0.64 |
| Recall (churn class) | 0.64 |
| Accuracy | 0.81 |
| Brier score | 0.144 |

The model catches about 64% of customers who actually churn, and about 64% of the customers it flags actually do churn — a reasonably balanced trade-off rather than one that over-targets or under-targets.

### Calibration

Checked with a 10-bin calibration curve on the held-out test set. The model is **well calibrated below ~40% predicted probability** — predicted and actual churn rates match closely. **Above ~40%, it is moderately overconfident**, a known effect of SMOTE resampling used to correct class imbalance during training. For example, a predicted 65% churn probability corresponds to an actual observed rate closer to 52%.

**Practical impact:** dollar figures shown in the app for high-risk customers (above ~50% predicted probability) may be optimistic by roughly 15-25%. The app surfaces this directly with an in-context warning when a customer's predicted probability exceeds 50%.

---

## ROI logic
