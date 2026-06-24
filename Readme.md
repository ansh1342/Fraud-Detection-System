# AI-Powered Fraud Detection System

An end-to-end Machine Learning project to detect fraudulent financial transactions using engineered transaction features and a Random Forest model, deployed with Streamlit and Docker.

---

## Project Overview

This project predicts whether a financial transaction is **fraudulent or legitimate** based on transaction details such as transaction type, amount, sender/receiver balances, and engineered behavioral features.

The project includes:
- Data preprocessing and EDA
- Feature engineering for fraud-specific transaction patterns
- Model training and evaluation
- Streamlit web app for interactive prediction
- Dockerized deployment

---

## Problem Statement

Financial fraud is a major challenge for banks, fintech companies, payment platforms, and digital wallets. Fraudulent transactions often follow unusual balance movement patterns, especially in transfer and cash-out operations.

The goal of this project is to build a machine learning model that can identify suspicious transactions with high recall and strong fraud detection performance.

---

## Dataset Features

Original dataset fields:
- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

---

## Feature Engineering

Additional engineered features used in the project:

- `balance_diff_orig`
- `balance_diff_dest`
- `hour`
- `day`
- `is_account_emptied`
- `full_balance_transfer`
- `amount_balance_ratio`
- `dest_balance_ratio`
- `sender_balance_missing`
- `receiver_balance_missing`
- `log_amount`
- `high_amount`

These features help the model capture suspicious fraud patterns beyond raw transaction values.

---

## Model Used

- **Random Forest Classifier**
- Pipeline-based preprocessing with:
  - `StandardScaler`
  - `OneHotEncoder`
  - `ColumnTransformer`

---

## Model Performance

Evaluation metrics achieved:

- **ROC-AUC:** ~0.9988
- **Cross-Validation F1 Score:** ~0.9975
- Strong fraud recall and precision on the test set

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Docker

---

## Project Structure

```bash
Fraud Detection Final/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── models/
│   └── fraud_model.pkl
└── data/