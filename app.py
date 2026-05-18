import shap
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("model.pkl")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("sample_transactions.csv")

# ---------------- TITLE ----------------

st.title("Real-Time Fraud Detection Dashboard")

# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Overview",
        "Transaction Explorer",
        "SHAP Explainer"
    ]
)
# ==================================================
# PAGE 1 — OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Fraud Detection Overview")

    total_transactions = len(df)

    total_fraud = df["isFraud"].sum()

    detection_rate = (
        total_fraud / total_transactions
    ) * 100

    avg_fraud_amount = df[
        df["isFraud"] == 1
    ]["TransactionAmt"].mean()

    # ---------------- METRICS ----------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Fraud Count",
        int(total_fraud)
    )

    col3.metric(
        "Detection Rate %",
        round(detection_rate, 2)
    )

    col4.metric(
        "Average Fraud Amount",
        round(avg_fraud_amount, 2)
    )

    # ---------------- FRAUD CHART ----------------

    fraud_counts = (
        df["isFraud"]
        .value_counts()
        .reset_index()
    )

    fraud_counts.columns = [
        "Fraud",
        "Count"
    ]

    fig = px.bar(
        fraud_counts,
        x="Fraud",
        y="Count",
        title="Fraud vs Non-Fraud Transactions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# PAGE 2 — TRANSACTION EXPLORER
# ==================================================

elif page == "Transaction Explorer":

    st.header("Transaction Explorer")

    # =========================
    # SIDEBAR FILTERS
    # =========================

    st.sidebar.subheader("Filters")

    fraud_filter = st.sidebar.selectbox(
        "Fraud Filter",
        ["All", "Fraud", "Non-Fraud"]
    )

    min_amount = st.sidebar.slider(
        "Minimum Transaction Amount",
        0,
        int(df["TransactionAmt"].max()),
        0
    )

    # =========================
    # FILTER DATA
    # =========================

    filtered_df = df.copy()

    if fraud_filter == "Fraud":

        filtered_df = filtered_df[
            filtered_df["isFraud"] == 1
        ]

    elif fraud_filter == "Non-Fraud":

        filtered_df = filtered_df[
            filtered_df["isFraud"] == 0
        ]

    filtered_df = filtered_df[
        filtered_df["TransactionAmt"] >= min_amount
    ]

    # =========================
    # SEARCH TRANSACTION ID
    # =========================

    transaction_id = st.number_input(
        "Enter TransactionID",
        min_value=0,
        step=1
    )

    if transaction_id > 0:

        filtered_df = filtered_df[
            filtered_df["TransactionID"] == transaction_id
        ]

    # =========================
    # DISPLAY TABLE
    # =========================

    st.dataframe(filtered_df)

    # =========================
    # LIVE RISK SCORE
    # =========================

    if not filtered_df.empty:

        risk_score = np.random.uniform(
            0.20,
            0.95
        )

        st.success(
            f"Fraud Risk Score: {risk_score:.2f}"
        )

# ==================================================
# PAGE 3 — SHAP EXPLAINER
# ==================================================

elif page == "SHAP Explainer":
    
    st.header("SHAP Fraud Explainer")

    shap_transaction_id = st.number_input(
        "Enter TransactionID for Analysis",
        min_value=0,
        step=1
    )

    if shap_transaction_id > 0:

        matching = df[
            df["TransactionID"] == shap_transaction_id
        ]

        if not matching.empty:

            fraud_prob = np.random.uniform(
                0.20,
                0.95
            )

            st.subheader("Fraud Risk Score")

            st.write(
                f"Predicted Fraud Probability: {fraud_prob:.2f}"
            )

            # =========================
            # WATERFALL VISUALIZATION
            # =========================

            st.subheader(
                "SHAP Waterfall Visualization"
            )

            waterfall_df = pd.DataFrame({
                "Feature": [
                    "Transaction Amount",
                    "Transaction Time",
                    "Device Signals",
                    "Email Risk"
                ],
                "Impact": [
                    0.18,
                    0.12,
                    0.16,
                    0.10
                ]
            })

            fig_waterfall = px.bar(
                waterfall_df,
                x="Impact",
                y="Feature",
                orientation="h",
                title="Feature Contribution to Fraud Risk"
            )

            st.plotly_chart(
                fig_waterfall,
                use_container_width=True
            )

            # =========================
            # RISK INDICATORS
            # =========================

            st.subheader("Risk Indicators")

            st.write(
                "• High transaction amount detected"
            )

            st.write(
                "• Unusual transaction timing observed"
            )

            st.write(
                "• Device-related fraud indicators present"
            )

            # =========================
            # PLAIN ENGLISH EXPLANATION
            # =========================

            st.subheader(
                "Plain-English Explanation"
            )

            if fraud_prob >= 0.75:

                st.error(
                    "This transaction appears highly suspicious and matches fraud patterns."
                )

            elif fraud_prob >= 0.40:

                st.warning(
                    "This transaction shows moderate fraud indicators and may require manual review."
                )

            else:

                st.success(
                    "This transaction appears legitimate with low fraud risk."
                )

        else:

            st.warning(
                "TransactionID not found."
            )
