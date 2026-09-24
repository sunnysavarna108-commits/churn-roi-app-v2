import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn ROI", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")


model = load_model()

# ---------------- Header ----------------
st.title("📉 Customer Churn & Retention ROI")
st.caption("Predict a customer's churn probability and see whether a retention campaign pays off.")

# ---------------- Sidebar ----------------
st.sidebar.title("Inputs")

with st.sidebar.expander("👤 Customer profile", expanded=True):
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=65.0)
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with st.sidebar.expander("📡 Services"):
    phone_service = st.selectbox("Phone service", ["Yes", "No"])
    if phone_service == "Yes":
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])
    else:
        multiple_lines = "No phone service"

    internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    if internet_service == "No":
        online_security = online_backup = device_protection = "No internet service"
        tech_support = streaming_tv = streaming_movies = "No internet service"
    else:
        online_security = st.selectbox("Online security", ["No", "Yes"])
        online_backup = st.selectbox("Online backup", ["No", "Yes"])
        device_protection = st.selectbox("Device protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])

with st.sidebar.expander("🧾 Account"):
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

with st.sidebar.expander("💰 Campaign economics", expanded=True):
    customer_ltv = st.number_input("Avg customer lifetime value ($)", min_value=0, value=1200)
    campaign_cost = st.number_input("Cost of retention campaign ($)", min_value=0, value=50)
    success_rate = st.slider("Expected campaign success rate (%)", 1, 100, 25) / 100.0
    threshold = st.slider("High-risk threshold (%)", 10, 90, 50) / 100.0

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
    "TotalCharges": tenure * monthly_charges,  # estimate
}])

churn_prob = float(model.predict_proba(customer_data)[0][1])

# ---------------- Financials ----------------
expected_saved = churn_prob * success_rate * customer_ltv
net_gain = expected_saved - campaign_cost
roi_pct = (net_gain / campaign_cost) * 100 if campaign_cost > 0 else 0
denom = churn_prob * customer_ltv
breakeven = (campaign_cost / denom) if denom > 0 else None

# ---------------- Risk level ----------------
if churn_prob >= threshold:
    risk_label, risk_icon = "High risk", "🔴"
elif churn_prob >= threshold * 0.6:
    risk_label, risk_icon = "Medium risk", "🟠"
else:
    risk_label, risk_icon = "Low risk", "🟢"

# ---------------- Layout ----------------
left, right = st.columns([1, 2], gap="large")

with left:
    with st.container(border=True):
        st.subheader("Churn prediction")
        st.metric("Probability of churn", f"{churn_prob:.1%}")
        st.progress(min(max(churn_prob, 0.0), 1.0))
        st.markdown(f"### {risk_icon} {risk_label}")
        st.caption(f"High-risk threshold: {threshold:.0%}")

with right:
    with st.container(border=True):
        st.subheader("Campaign ROI")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Campaign cost", f"${campaign_cost:,.0f}")
        c2.metric("Expected value saved", f"${expected_saved:,.2f}")
        c3.metric("Net gain", f"${net_gain:,.2f}", delta=f"{net_gain:,.2f}")
        c4.metric("ROI", f"{roi_pct:.0f}%")

        if net_gain > 0 and churn_prob >= threshold:
            st.success("✅ Recommended: run the retention campaign for this customer.")
        elif net_gain > 0:
            st.info("ℹ️ Campaign is profitable, but risk is below your threshold. Optional.")
        else:
            st.warning("⚠️ Not recommended: expected value saved is below the campaign cost.")

        if breakeven is not None:
            if breakeven <= 1:
                st.caption(f"Break-even success rate: **{breakeven:.1%}** (campaign pays off above this).")
            else:
                st.caption("Break-even success rate is above 100%: this campaign can't pay off for this customer.")

st.subheader("Cost vs. expected value")
chart_df = pd.DataFrame(
    {"Amount ($)": [campaign_cost, expected_saved]},
    index=["Campaign cost", "Expected value saved"],
)
st.bar_chart(chart_df, horizontal=True)

with st.expander("🔍 Model input (for verification)"):
    st.dataframe(customer_data, hide_index=True)
    st.caption("TotalCharges is estimated as tenure × monthly charges.")
