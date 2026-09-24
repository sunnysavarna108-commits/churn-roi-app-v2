import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn ROI", page_icon="📉", layout="wide")

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")


model = load_model()

# ---------------- Header ----------------
st.title("📉 Customer Churn & Retention ROI")
st.caption("Predict churn probability and see whether a retention campaign pays off.")

# ---------------- Sidebar ----------------
st.sidebar.title("Inputs")

with st.sidebar.expander("💰 Campaign economics (used by both tabs)", expanded=True):
    customer_ltv = st.number_input("Avg customer lifetime value ($)", min_value=0, value=1200)
    campaign_cost = st.number_input("Cost of retention campaign ($)", min_value=0, value=50)
    success_rate = st.slider("Expected campaign success rate (%)", 1, 100, 25) / 100.0
    threshold = st.slider("High-risk threshold (%)", 10, 90, 50) / 100.0

with st.sidebar.expander("👤 Customer profile (single customer tab)", expanded=True):
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=65.0)
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with st.sidebar.expander("📡 Services (single customer tab)"):
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

with st.sidebar.expander("🧾 Account (single customer tab)"):
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )


# ---------------- Helpers ----------------
def risk_level(p):
    if p >= threshold:
        return "🔴 High risk"
    if p >= threshold * 0.6:
        return "🟠 Medium risk"
    return "🟢 Low risk"


def clean_batch(df):
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    # SeniorCitizen may be Yes/No text or 0/1
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({"Yes": 1, "No": 0})
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # blanks become NaN, pipeline imputes
    return df


tab_single, tab_batch = st.tabs(["🧍 Single customer", "📁 Batch scoring (CSV)"])

# ================= TAB 1: single customer =================
with tab_single:
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

    expected_saved = churn_prob * success_rate * customer_ltv
    net_gain = expected_saved - campaign_cost
    roi_pct = (net_gain / campaign_cost) * 100 if campaign_cost > 0 else 0
    denom = churn_prob * customer_ltv
    breakeven = (campaign_cost / denom) if denom > 0 else None

    left, right = st.columns([1, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Churn prediction")
            st.metric("Probability of churn", f"{churn_prob:.1%}")
            st.progress(min(max(churn_prob, 0.0), 1.0))
            st.markdown(f"### {risk_level(churn_prob)}")
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

# ================= TAB 2: batch scoring =================
with tab_batch:
    st.subheader("Score many customers at once")
    st.write(
        "Upload a CSV with these columns (extra columns like `customerID` or `Churn` are ignored):"
    )
    st.code(", ".join(REQUIRED_COLUMNS), language=None)

    template = pd.DataFrame([
        ["Female", 0, "No", "No", 3, "Yes", "No", "Fiber optic", "No", "No", "No", "No", "No", "No",
         "Month-to-month", "Yes", "Electronic check", 85.0, 255.0],
        ["Male", 0, "Yes", "Yes", 60, "Yes", "Yes", "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No",
         "Two year", "No", "Credit card (automatic)", 55.0, 3300.0],
    ], columns=REQUIRED_COLUMNS)
    st.download_button(
        "⬇️ Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="customer_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        data = clean_batch(raw[REQUIRED_COLUMNS])

        try:
            probs = model.predict_proba(data)[:, 1]
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            st.stop()

        result = raw.copy()
        result["churn_probability"] = probs
        result["risk_level"] = [risk_level(p) for p in probs]
        result["expected_value_saved"] = probs * success_rate * customer_ltv
        result["net_gain"] = result["expected_value_saved"] - campaign_cost
        result["target_campaign"] = (result["churn_probability"] >= threshold) & (result["net_gain"] > 0)

        targeted = result[result["target_campaign"]]
        total_cost = len(targeted) * campaign_cost
        total_saved = targeted["expected_value_saved"].sum()
        total_net = total_saved - total_cost
        total_roi = (total_net / total_cost * 100) if total_cost > 0 else 0

        st.markdown("### Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers scored", f"{len(result):,}")
        m2.metric("High risk", f"{(result['churn_probability'] >= threshold).sum():,}")
        m3.metric("Average churn probability", f"{result['churn_probability'].mean():.1%}")
        m4.metric("Customers to target", f"{len(targeted):,}")

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Total campaign cost", f"${total_cost:,.0f}")
        n2.metric("Total expected value saved", f"${total_saved:,.0f}")
        n3.metric("Total net gain", f"${total_net:,.0f}")
        n4.metric("Overall ROI", f"{total_roi:.0f}%")
        st.caption(
            "Only customers at or above the high-risk threshold whose expected value saved exceeds the "
            "campaign cost are targeted."
        )

        st.markdown("#### Risk distribution")
        st.bar_chart(result["risk_level"].value_counts())

        st.markdown("#### Customers ranked by churn probability")
        shown = result.sort_values("churn_probability", ascending=False)
        st.dataframe(
            shown,
            hide_index=True,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "churn_probability", min_value=0.0, max_value=1.0, format="%.1f%%"
                    if False else "percent"
                ),
                "expected_value_saved": st.column_config.NumberColumn(format="$%.2f"),
                "net_gain": st.column_config.NumberColumn(format="$%.2f"),
            },
        )

        st.download_button(
            "⬇️ Download scored results (CSV)",
            shown.to_csv(index=False).encode("utf-8"),
            file_name="scored_customers.csv",
            mime="text/csv",
        )
