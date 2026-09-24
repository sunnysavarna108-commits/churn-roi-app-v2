import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn ROI", layout="wide")
st.title("Customer Churn & Retention ROI")
st.markdown("Predict churn probability and evaluate the financial return of retention campaigns.")

st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=65.0)

st.sidebar.header("Campaign Economics")
customer_ltv = st.sidebar.number_input("Avg Customer Lifetime Value ($)", min_value=0, value=1200)
campaign_cost = st.sidebar.number_input("Cost of Retention Campaign ($)", min_value=0, value=50)
success_rate = st.sidebar.slider("Expected Campaign Success Rate (%)", 1, 100, 25) / 100.0

# --- Model Integration ---
# Uncomment the block below and ensure your model file is in the same repository

@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")
    
model = load_model()
customer_data = pd.DataFrame({"tenure": [tenure], "MonthlyCharges": [monthly_charges]})
churn_prob = model.predict_proba(customer_data)[0][1]


# Mock probability for structural demonstration (Delete this once your model is linked)
#churn_prob = np.clip((80 - tenure + (monthly_charges * 0.1)) / 100, 0.05, 0.95)

st.subheader("Prediction")
st.metric("Probability of Churn", f"{churn_prob:.1%}")

st.subheader("Financial Impact Analysis")
if churn_prob > 0.5:
    st.warning("High Churn Risk: Retention intervention recommended.")
    
    expected_saved_value = churn_prob * success_rate * customer_ltv
    net_roi = expected_saved_value - campaign_cost
    roi_percentage = (net_roi / campaign_cost) * 100 if campaign_cost > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Intervention Cost", f"${campaign_cost:,.2f}")
    col2.metric("Expected Value Saved", f"${expected_saved_value:,.2f}")
    col3.metric("Estimated ROI", f"{roi_percentage:.1f}%")
else:
    st.success("Low Churn Risk: No immediate intervention required.")
