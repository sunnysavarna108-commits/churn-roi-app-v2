import pandas as pd
from logic import (
    risk_level,
    expected_value_saved,
    net_gain,
    roi_percentage,
    breakeven_success_rate,
    portfolio_profit_at_threshold,
    clean_batch,
)


def test_risk_level_boundaries():
    assert risk_level(0.8, threshold=0.5) == "🔴 High risk"
    assert risk_level(0.5, threshold=0.5) == "🔴 High risk"
    assert risk_level(0.35, threshold=0.5) == "🟠 Medium risk"
    assert risk_level(0.1, threshold=0.5) == "🟢 Low risk"


def test_expected_value_saved_is_zero_when_prob_is_zero():
    assert expected_value_saved(prob=0, success_rate=0.25, ltv=1200) == 0


def test_expected_value_saved_scales_linearly():
    low = expected_value_saved(0.5, 0.25, 1000)
    high = expected_value_saved(1.0, 0.25, 1000)
    assert high == 2 * low


def test_net_gain_negative_when_cost_exceeds_value():
    gain = net_gain(prob=0.1, success_rate=0.1, ltv=100, cost=50)
    assert gain < 0


def test_net_gain_positive_when_value_exceeds_cost():
    gain = net_gain(prob=0.9, success_rate=0.5, ltv=1000, cost=50)
    assert gain > 0


def test_roi_percentage_zero_cost_does_not_crash():
    assert roi_percentage(prob=0.5, success_rate=0.5, ltv=1000, cost=0) == 0


def test_roi_percentage_matches_manual_calc():
    prob, rate, ltv, cost = 0.6, 0.25, 1200, 50
    expected = ((prob * rate * ltv - cost) / cost) * 100
    assert round(roi_percentage(prob, rate, ltv, cost), 4) == round(expected, 4)


def test_breakeven_success_rate_known_value():
    be = breakeven_success_rate(prob=0.64, ltv=1200, cost=50)
    assert round(be, 3) == round(50 / (0.64 * 1200), 3)


def test_breakeven_success_rate_none_when_prob_zero():
    assert breakeven_success_rate(prob=0, ltv=1200, cost=50) is None


def test_portfolio_profit_at_threshold_excludes_below_threshold():
    df = pd.DataFrame({
        "churn_probability": [0.9, 0.8, 0.3, 0.1],
        "expected_value_saved": [500, 400, 100, 20],
    })
    profit, n = portfolio_profit_at_threshold(df, t=0.5, cost=50)
    assert n == 2
    assert profit == (500 + 400) - (2 * 50)


def test_portfolio_profit_at_threshold_zero_matches_none_above():
    df = pd.DataFrame({"churn_probability": [0.1], "expected_value_saved": [10]})
    profit, n = portfolio_profit_at_threshold(df, t=0.99, cost=50)
    assert n == 0
    assert profit == 0


def test_clean_batch_converts_senior_citizen_text():
    df = pd.DataFrame({
        "SeniorCitizen": ["Yes", "No"],
        "tenure": ["12", "24"],
        "MonthlyCharges": ["65.0", "80.0"],
        "TotalCharges": ["780", ""],
    })
    cleaned = clean_batch(df)
    assert cleaned["SeniorCitizen"].tolist() == [1, 0]
    assert cleaned["tenure"].tolist() == [12, 24]
    assert pd.isna(cleaned["TotalCharges"].iloc[1])
