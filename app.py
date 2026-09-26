import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import xgboost as xgb
import joblib

st.set_page_config(page_title="Customer Churn ROI", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    h1 { font-weight: 800; letter-spacing: -0.5px; }

    [data-testid="stMetric"] {
        background: transparent;
        border: none;
        padding: 4px 0;
    }
    [data-testid="stMetricLabel"] {
        opacity: 0.75;
        white-space: normal;
        overflow: visible;
        text-overflow: unset;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        white-space: normal;
        overflow: visible;
        text-overflow: unset;
    }

    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
    button[data-baseweb="tab"] { font-size: 1rem; padding: 10px 18px; }
    [data-testid="stSidebar"] { border-right: 1px solid #2A3142; }
    .stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

WHAT_IF_OPTIONS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
    "PaperlessBilling": ["Yes", "No"],
    "OnlineSecurity": ["No", "Yes"],
    "TechSupport": ["No", "Yes"],
}


@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")


model = load_model()

# ---------------- Header ----------------
st.title("📉 Customer Churn & Retention ROI")
st.caption("Predict churn probability and see whether a retention campaign pays off.")

# ---------------- Sidebar ----------------
st.sidebar.title("Inputs")

with st.sidebar.expander("💰 Campaign economics (used by all tabs)", expanded=True):
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
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({"Yes": 1, "No": 0})
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def original_column(name):
    name = name.split("__", 1)[1] if "__" in name else name
    matches = [c for c in REQUIRED_COLUMNS if name == c or name.startswith(c + "_")]
    return max(matches, key=len) if matches else name


@st.cache_data
def get_feature_importance():
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    names = pre.get_feature_names_out()
    values = clf.feature_importances_
    if len(names) != len(values):
        return None
    df = pd.DataFrame({"Feature": [original_column(n) for n in names], "Importance": values})
    df = df.groupby("Feature", as_index=False)["Importance"].sum()
    df["Importance"] = df["Importance"] / df["Importance"].sum()
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


tab_single, tab_batch, tab_importance, tab_why = st.tabs(
    ["🧍 Single customer", "📁 Batch scoring (CSV)", "📊 What drives churn", "🔎 Why this score"]
)

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
        "TotalCharges": tenure * monthly_charges,
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
            st.metric("Churn probability", f"{churn_prob:.1%}")
            st.progress(min(max(churn_prob, 0.0), 1.0))
            st.markdown(f"### {risk_level(churn_prob)}")
            st.caption(f"High-risk threshold: {threshold:.0%}")

    with right:
        with st.container(border=True):
            st.subheader("Campaign ROI")

            r1c1, r1c2 = st.columns(2)
            r1c1.metric("Cost", f"${campaign_cost:,.0f}")
            r1c2.metric("Value saved", f"${expected_saved:,.0f}")

            r2c1, r2c2 = st.columns(2)
            r2c1.metric("Net gain", f"${net_gain:,.0f}")
            r2c2.metric("ROI", f"{roi_pct:.0f}%")

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

    st.subheader("Net gain vs. campaign success rate")
    st.caption("Shows how net gain changes if the real success rate turns out higher or lower than your estimate.")

    rates = np.linspace(0.01, 1.0, 100)
    sens_df = pd.DataFrame({
        "Success rate": rates,
        "Net gain": churn_prob * rates * customer_ltv - campaign_cost,
    })

    line = alt.Chart(sens_df).mark_line(color="#4F8BF9", strokeWidth=3).encode(
        x=alt.X("Success rate", axis=alt.Axis(format="%"), title="Campaign success rate"),
        y=alt.Y("Net gain", title="Net gain ($)"),
        tooltip=[alt.Tooltip("Success rate", format=".0%"), alt.Tooltip("Net gain", format="$.2f")],
    )
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y")

    layers = [line, zero_line]

    if breakeven is not None and 0 < breakeven <= 1:
        be_point = alt.Chart(pd.DataFrame({"x": [breakeven], "y": [0]})).mark_point(
            color="#FF4B4B", size=100
        ).encode(x="x", y="y")
        be_rule = alt.Chart(pd.DataFrame({"x": [breakeven]})).mark_rule(
            color="#FF4B4B", strokeDash=[4, 4]
        ).encode(x="x")
        layers += [be_rule, be_point]

    current_point = alt.Chart(pd.DataFrame({"x": [success_rate], "y": [net_gain]})).mark_point(
        color="#00C48C", size=140, shape="diamond"
    ).encode(x="x", y="y")
    layers.append(current_point)

    st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
    st.caption(
        "🔴 Red = break-even point · 🟢 Green diamond = your current setting. "
        "Above the dashed gray line, the campaign is profitable."
    )

    with st.expander("🔍 Model input (for verification)"):
        st.dataframe(customer_data, hide_index=True)
        st.caption("TotalCharges is estimated as tenure × monthly charges.")

# ================= TAB 2: batch scoring =================
with tab_batch:
    st.subheader("Score many customers at once")
    st.write("Upload a CSV with these columns (extra columns like `customerID` or `Churn` are ignored):")
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
        m1.metric("Scored", f"{len(result):,}")
        m2.metric("High risk", f"{(result['churn_probability'] >= threshold).sum():,}")
        m3.metric("Avg churn %", f"{result['churn_probability'].mean():.0%}")
        m4.metric("To target", f"{len(targeted):,}")

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Cost", f"${total_cost:,.0f}")
        n2.metric("Value saved", f"${total_saved:,.0f}")
        n3.metric("Net gain", f"${total_net:,.0f}")
        n4.metric("ROI", f"{total_roi:.0f}%")
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
                    "churn_probability", min_value=0.0, max_value=1.0, format="percent"
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

# ================= TAB 3: feature importance =================
with tab_importance:
    st.subheader("What drives churn in this model")
    st.write(
        "Relative importance of each customer attribute in the XGBoost model. "
        "Higher means the model relies on that attribute more when splitting customers into churn / stay."
    )

    try:
        imp = get_feature_importance()
    except Exception as e:
        imp = None
        st.error(f"Could not read feature importance from the model: {e}")

    if imp is not None:
        top_n = st.slider("Number of features to show", 5, len(imp), min(10, len(imp)))
        top = imp.head(top_n)

        st.bar_chart(top, x="Feature", y="Importance", horizontal=True, sort="-Importance")

        top3 = ", ".join(top["Feature"].head(3))
        st.info(f"Top drivers: **{top3}**. Retention efforts are likely to matter most for customers with risky values here.")

        with st.expander("See full table"):
            st.dataframe(
                imp,
                hide_index=True,
                column_config={"Importance": st.column_config.NumberColumn(format="percent")},
            )

        st.caption(
            "Importance shows what the model uses, not what causes churn. "
            "Use it as a guide for where to investigate, not as proof of cause."
        )
    else:
        st.warning("Feature names and importances didn't line up, so the chart can't be drawn.")

# ================= TAB 4: why this score (SHAP) + what-if =================
with tab_why:
    st.subheader("Why this customer got this score")
    st.write(
        "Each bar shows how much a customer attribute pushed the churn risk up "
        "or down compared with an average customer. Uses the customer from the sidebar."
    )

    try:
        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]

        X_t = pre.transform(customer_data)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        X_t = np.asarray(X_t, dtype=float)

        contrib = clf.get_booster().predict(xgb.DMatrix(X_t), pred_contribs=True)[0]
        values = contrib[:-1]
        names = pre.get_feature_names_out()

        df_c = pd.DataFrame({"Feature": [original_column(n) for n in names], "Impact": values})
        df_c = df_c.groupby("Feature", as_index=False)["Impact"].sum()
        df_c = df_c.reindex(df_c["Impact"].abs().sort_values(ascending=False).index).head(10)
        df_c["Direction"] = np.where(df_c["Impact"] > 0, "Raises churn risk", "Lowers churn risk")

        st.bar_chart(df_c, x="Feature", y="Impact", color="Direction", horizontal=True, sort="-Impact")

        top_up = df_c[df_c["Impact"] > 0].head(2)["Feature"].tolist()
        top_down = df_c[df_c["Impact"] < 0].head(2)["Feature"].tolist()
        if top_up:
            st.error(f"Biggest risk drivers: **{', '.join(top_up)}**")
        if top_down:
            st.success(f"Biggest protective factors: **{', '.join(top_down)}**")

        st.caption(
            "Values are in log-odds units (the model's internal scale), not percentage points. "
            "Positive means higher churn risk. These explain the model's behaviour, not causes of churn."
        )
    except Exception as e:
        st.error(f"Could not compute explanations: {e}")

    st.divider()

    # ----------------
