"""Pure business-logic functions for the churn ROI app — no Streamlit dependency,
so these can be imported and tested independently."""

import pandas as pd


def risk_level(prob, threshold):
    if prob >= threshold:
        return "🔴 High risk"
    if prob >= threshold * 0.6:
        return "🟠 Medium risk"
    return "🟢 Low risk"


def expected_value_saved(prob, success_rate, ltv):
    return prob * success_rate * ltv


def net_gain(prob, success_rate, ltv, cost):
    return expected_value_saved(prob, success_rate, ltv) - cost


def roi_percentage(prob, success_rate, ltv, cost):
    gain = net_gain(prob, success_rate, ltv, cost)
    return (gain / cost) * 100 if cost > 0 else 0


def breakeven_success_rate(prob, ltv, cost):
    denom = prob * ltv
    return (cost / denom) if denom > 0 else None


def portfolio_profit_at_threshold(df, t, cost, prob_col="churn_probability", saved_col="expected_value_saved"):
    """Total net profit if every customer at/above threshold t is targeted."""
    mask = df[prob_col] >= t
    n = int(mask.sum())
    total_cost = n * cost
    total_saved = df.loc[mask, saved_col].sum()
    return total_saved - total_cost, n


def clean_batch(df):
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({"Yes": 1, "No": 0})
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
