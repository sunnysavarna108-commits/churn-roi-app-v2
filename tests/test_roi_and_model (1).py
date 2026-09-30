"""Tests for logic.py (ROI maths, batch helpers) and the deployed model.

Put this file in the `tests/` folder and run `pytest -q` from the project root.
"""
import os

import pandas as pd
import pytest

from logic import (
    risk_level,
    expected_value_saved,
    net_gain,
    roi_percentage,
    breakeven_success_rate,
    portfolio_profit_at_threshold,
    clean_batch,
)

# ---------------------------------------------------------------- ROI maths

def test_expected_value_saved_matches_formula():
    # 0.64 churn prob x 25% success x $1200 LTV = $192 (the value shown in the app)
    assert expected_value_saved(0.64, 0.25, 1200) == pytest.approx(192.0)


def test_expected_value_saved_zero_when_no_churn_risk():
    assert expected_value_saved(0.0, 0.25, 1200) == pytest.approx(0.0)


def test_net_gain_is_value_saved_minus_cost():
    assert net_gain(0.64, 0.25, 1200, 50) == pytest.approx(192.0 - 50.0)


def test_net_gain_negative_when_risk_is_zero():
    # No churn risk means the campaign cost is pure loss
    assert net_gain(0.0, 0.25, 1200, 50) == pytest.approx(-50.0)


def test_roi_percentage_is_net_gain_over_cost():
    # net gain $142 on $50 cost = 284%
    assert roi_percentage(0.64, 0.25, 1200, 50) == pytest.approx(284.0)


def test_roi_is_zero_when_cost_is_zero():
    # Avoids division by zero
    assert roi_percentage(0.64, 0.25, 1200, 0) == 0


def test_net_gain_increases_with_churn_probability():
    assert net_gain(0.60, 0.25, 1200, 50) > net_gain(0.20, 0.25, 1200, 50)


def test_net_gain_increases_with_success_rate():
    assert net_gain(0.50, 0.40, 1200, 50) > net_gain(0.50, 0.10, 1200, 50)


# ---------------------------------------------------------------- break-even

def test_breakeven_success_rate_matches_formula():
    # cost / (churn_prob x LTV) = 50 / (0.64 x 1200) = 6.5%
    assert breakeven_success_rate(0.64, 1200, 50) == pytest.approx(50 / (0.64 * 1200))


def test_breakeven_is_none_when_churn_probability_is_zero():
    assert breakeven_success_rate(0.0, 1200, 50) is None


def test_net_gain_is_zero_at_breakeven_success_rate():
    p, ltv, cost = 0.40, 1200, 50
    be = breakeven_success_rate(p, ltv, cost)
    assert net_gain(p, be, ltv, cost) == pytest.approx(0.0, abs=1e-6)


def test_net_gain_is_zero_at_breakeven_churn_probability():
    # Default settings: cost / (success_rate x LTV) = 50 / 300, about 16.7%
    be_prob = 50 / (0.25 * 1200)
    assert be_prob == pytest.approx(0.1667, abs=1e-3)
    assert net_gain(be_prob, 0.25, 1200, 50) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- risk level

def test_risk_level_high_at_or_above_threshold():
    assert "High" in risk_level(0.90, 0.50)
    assert "High" in risk_level(0.50, 0.50)  # boundary counts as high


def test_risk_level_medium_between_60_percent_of_threshold_and_threshold():
    # threshold 0.5 -> medium band starts at 0.3
    assert "Medium" in risk_level(0.35, 0.50)


def test_risk_level_low_below_medium_band():
    assert "Low" in risk_level(0.01, 0.50)


# ---------------------------------------------------------------- portfolio profit

@pytest.fixture
def scored():
    return pd.DataFrame({
        "churn_probability": [0.90, 0.50, 0.10],
        "expected_value_saved": [100.0, 60.0, 5.0],
    })


def test_portfolio_profit_targets_customers_at_or_above_threshold(scored):
    profit, n = portfolio_profit_at_threshold(scored, 0.50, 20)
    assert n == 2                                  # 0.90 and 0.50
    assert profit == pytest.approx(160.0 - 40.0)   # saved minus 2 x cost


def test_portfolio_profit_at_zero_threshold_targets_everyone(scored):
    profit, n = portfolio_profit_at_threshold(scored, 0.0, 20)
    assert n == 3
    assert profit == pytest.approx(165.0 - 60.0)


def test_portfolio_profit_is_zero_when_nobody_is_targeted(scored):
    profit, n = portfolio_profit_at_threshold(scored, 1.0, 20)
    assert n == 0
    assert profit == pytest.approx(0.0)


# ---------------------------------------------------------------- clean_batch

def test_clean_batch_converts_types_and_strips_whitespace():
    raw = pd.DataFrame({
        "gender": [" Male ", "Female"],
        "SeniorCitizen": ["Yes", "No"],
        "tenure": [12, 5],
        "MonthlyCharges": [65.0, 80.0],
        "TotalCharges": ["780.0", " "],   # blank string, as in the Telco file
    })
    out = clean_batch(raw)

    assert out["gender"].tolist() == ["Male", "Female"]
    assert out["SeniorCitizen"].tolist() == [1, 0]
    assert out["TotalCharges"].iloc[0] == pytest.approx(780.0)
    assert pd.isna(out["TotalCharges"].iloc[1])     # blank becomes NaN
    assert raw["gender"].iloc[0] == " Male "        # input is not modified


# ---------------------------------------------------------------- deployed model

MODEL_FILE = "churn_pipeline_calibrated_weighted.pkl"

CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 65.0, "TotalCharges": 780.0,
}


TRAINED_SKLEARN_VERSION = "1.6.1"   # version the model was pickled with


@pytest.fixture(scope="module")
def model():
    joblib = pytest.importorskip("joblib")
    sklearn = pytest.importorskip("sklearn")
    if sklearn.__version__ != TRAINED_SKLEARN_VERSION:
        pytest.skip(
            f"Model was trained with scikit-learn {TRAINED_SKLEARN_VERSION}, "
            f"but {sklearn.__version__} is installed (pickles don't load across versions)"
        )
    if not os.path.exists(MODEL_FILE):
        pytest.skip(f"{MODEL_FILE} not found (run pytest from the project root)")
    return joblib.load(MODEL_FILE)


def test_model_probability_is_between_0_and_1(model):
    p = float(model.predict_proba(pd.DataFrame([CUSTOMER]))[0][1])
    assert 0.0 <= p <= 1.0


def test_model_scores_a_small_batch(model):
    batch = pd.DataFrame([CUSTOMER, {**CUSTOMER, "Contract": "Two year", "tenure": 60}])
    probs = model.predict_proba(batch)[:, 1]
    assert len(probs) == 2
    assert ((probs >= 0) & (probs <= 1)).all()


def test_two_year_contract_is_lower_risk_than_month_to_month(model):
    month = float(model.predict_proba(pd.DataFrame([CUSTOMER]))[0][1])
    two_year = float(
        model.predict_proba(pd.DataFrame([{**CUSTOMER, "Contract": "Two year"}]))[0][1]
    )
    assert two_year < month
