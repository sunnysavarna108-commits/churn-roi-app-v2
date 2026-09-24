import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn ROI", layout="wide")
st.title("Customer Churn & Retention ROI")
st.markdown("Predict churn probability and evaluate the financial return of retention campaigns.")


@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")


model = load_model()

# ---------------- Sidebar: customer profile ----------------
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=65.0)

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

st.sidebar.subheader("Services")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
if phone_service == "Yes":
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
else:
    multiple_lines = "No phone service"

internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
if internet_service == "No":
    online_security = online_backup = device_protection = "No internet service"
    tech_support = streaming_tv = streaming_movies = "No internet service"
else:
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])

st.sidebar.subheader("Account")
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)

# ---------------- Sidebar: campaign economics ----------------
st.sidebar.header("Campaign Economics")
customer_ltv = st.sidebar.number_input("Avg Customer Lifetime Value ($)", min_value=0, value=1200)
campaign_cost = st.sidebar.number_input("Cost of Retention Campaign ($)", min_value=0, value=50)
success_rate = st.sidebar.slider("Expected Campaign Success Rate (%)", 1, 100, 25) / 100.0

# ---------------- Prediction ----------------
customer_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": tenure * monthly_charges,  # estimate: tenure x monthly charges
}])

churn_prob = float(model.predict_proba(customer_data)[0][1])

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

with st.expander("Model input (for verification)"):
    st.dataframe(customer_data)
