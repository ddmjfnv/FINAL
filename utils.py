import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data(csv_path="data/cleaned_ai_solutions_web_server_logs.csv"):
    if not os.path.exists(csv_path):
        st.error(f"Data file not found at {csv_path}")
        return pd.DataFrame()  # Return empty DataFrame if file is missing

    df = pd.read_csv(csv_path)

    # Optional: preprocess datetime, drop NAs, format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'],  format='%H:%M:%S', errors='coerce').dt.hour.fillna(0).astype(int)

    return df
