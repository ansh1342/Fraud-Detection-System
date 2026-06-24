import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

model = joblib.load("models/fraud_detection_pipeline.pkl")

st.title("🛡️ AI-Powered Fraud Detection System")
st.write("Detect whether a financial transaction is fraudulent using Machine Learning.")

st.divider()

st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=5000.0
    )

with col2:
    newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=4000.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=10000.0
    )

    newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=11000.0
    )

predict_btn = st.button("🔍 Predict Fraud")

if predict_btn:

    step = 1

    balance_diff_orig = oldbalanceOrg - newbalanceOrig
    balance_diff_dest = newbalanceDest - oldbalanceDest

    hour = step % 24
    day = step // 24

    is_account_emptied = int(
        oldbalanceOrg > 0 and newbalanceOrig == 0
    )

    full_balance_transfer = int(
        amount >= oldbalanceOrg and oldbalanceOrg > 0
    )

    amount_balance_ratio = amount / (oldbalanceOrg + 1)

    dest_balance_ratio = amount / (oldbalanceDest + 1)

    sender_balance_missing = int(
        oldbalanceOrg == 0 and newbalanceOrig == 0
    )

    receiver_balance_missing = int(
        oldbalanceDest == 0 and newbalanceDest == 0
    )

    log_amount = np.log1p(amount)

    high_amount = int(amount > 200000)

    input_df = pd.DataFrame({
        "step": [step],
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
        "balance_diff_orig": [balance_diff_orig],
        "balance_diff_dest": [balance_diff_dest],
        "hour": [hour],
        "day": [day],
        "is_account_emptied": [is_account_emptied],
        "full_balance_transfer": [full_balance_transfer],
        "amount_balance_ratio": [amount_balance_ratio],
        "dest_balance_ratio": [dest_balance_ratio],
        "sender_balance_missing": [sender_balance_missing],
        "receiver_balance_missing": [receiver_balance_missing],
        "log_amount": [log_amount],
        "high_amount": [high_amount]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Prediction",
            "Fraud" if prediction == 1 else "Legitimate"
        )

    st.divider()

    if prediction == 1:
        st.error(
            f"⚠️ Fraud Detected\n\nFraud Probability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Legitimate Transaction\n\nFraud Probability: {probability:.2%}"
        )

    with st.expander("View Engineered Features"):
        st.dataframe(input_df)