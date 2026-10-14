# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:26:19 2026

@author: Nosike
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import time

# Page Configuration
st.set_page_config(page_title="Low-Resource CNN Fraud Dashboard", layout="wide")

# 1. Simulate or Load Lightweight 1D-CNN Model
@st.cache_resource
def load_fraud_cnn_model():
    # Compact architecture designed for low memory and CPU footprint
    model = Sequential([
        Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(5, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid') # Fraud probability score
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_fraud_cnn_model()

# 2. UI Layout - Header
st.title("🛡️ Low-Resource Banking Fraud Detection Dashboard")
st.markdown("Real-time transaction stream monitoring powered by lightweight 1D-CNN feature extraction.")

# 3. Sidebar Controls for Low-Resource Configuration
st.sidebar.header("⚙️ System Controls")
threshold = st.sidebar.slider("Fraud Alert Threshold", 0.0, 1.0, 0.75, 0.05)
refresh_rate = st.sidebar.selectbox("Stream Batch Interval (s)", [1, 2, 5], index=0)

# 4. Generate Simulated Low-Bandwidth Transaction Stream
@st.cache_data
def generate_mock_stream(n=50):
    data = {
        "transaction_id": [f"TXN-{np.random.randint(10000,99999)}" for _ in range(n)],
        "account_id": [f"ACC-{np.random.randint(1000,9999)}" for _ in range(n)],
        "amount_usd": np.random.exponential(scale=150, size=n).round(2),
        "distance_from_home_km": np.random.uniform(0, 50, size=n).round(1),
        "hour_of_day": np.random.randint(0, 24, size=n),
        "velocity_last_hour": np.random.poisson(lam=1.5, size=n),
        "device_risk_score": np.random.uniform(0, 1, size=n).round(2)
    }
    return pd.DataFrame(data)

df_stream = generate_mock_stream()

# 5. Process Features through CNN
def score_transactions(df):
    features = df[['amount_usd', 'distance_from_home_km', 'hour_of_day', 'velocity_last_hour', 'device_risk_score']].values
    # Reshape to 3D tensor for Conv1D: (Samples, Timesteps=5, Features=1)
    X = np.expand_dims(features, axis=2)
    # Predict probabilities
    preds = model.predict(X, verbose=0).flatten()
    df['fraud_score'] = preds
    df['status'] = np.where(df['fraud_score'] >= threshold, 'REVIEW REQUIRED', 'APPROVED')
    return df

scored_df = score_transactions(df_stream)

# 6. Key Metrics Overview
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions Monitored", len(scored_df))
col2.metric("Flagged Anomalies", len(scored_df[scored_df['status'] == 'REVIEW REQUIRED']))
col3.metric("System Load / CPU Est.", "Low (<4%)")

st.markdown("---")

# 7. Interactive Data Grid & Alerts
st.subheader("🚨 Live Flagged Queue (Analyst Action Required)")
flagged_q = scored_df[scored_df['status'] == 'REVIEW REQUIRED']

if not flagged_q.empty:
    st.dataframe(flagged_q[['transaction_id', 'account_id', 'amount_usd', 'fraud_score', 'status']], use_container_width=True)
    selected_txn = st.selectbox("Select Transaction ID to Resolve", flagged_q['transaction_id'])
    if st.button("Confirm Freeze Account / Block TXN"):
        st.success(f"Transaction {selected_txn} blocked successfully. Audit log updated.")
else:
    st.info("No high-risk transactions breaching the current threshold.")

st.subheader("📊 Full Transaction Log Stream")
st.dataframe(scored_df, use_container_width=True)
