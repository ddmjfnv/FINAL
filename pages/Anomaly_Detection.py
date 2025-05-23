import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Anomaly Detection", layout="wide")

# --- Authentication check ---
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.error("You must log in to access this page.")
    st.stop()

col1, col2 = st.columns([6, 1])  # Adjust widths as needed

with col2:
    if st.button("🔓 Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.rerun()

st.subheader("📉 Sales Anomaly Detection")
st.markdown("This page analyzes historical sales data to detect unusual spikes or drops in performance.")

# --- Simulate data (replace with your actual data) ---
np.random.seed(42)
dates = pd.date_range(end=datetime.today(), periods=100).to_pydatetime().tolist()
sales = np.random.normal(loc=5000, scale=1500, size=100).astype(int)
sales[15] = 15000  # High anomaly
sales[33] = 800    # Low anomaly

df = pd.DataFrame({"Date": dates, "Sales": sales})
df["Rolling_Mean"] = df["Sales"].rolling(window=7).mean()
df["Rolling_Std"] = df["Sales"].rolling(window=7).std()
df["Anomaly"] = ((df["Sales"] > df["Rolling_Mean"] + 2 * df["Rolling_Std"]) |
                 (df["Sales"] < df["Rolling_Mean"] - 2 * df["Rolling_Std"]))

# --- Visualization ---
fig = px.line(df, x="Date", y="Sales", title="Sales Over Time with Anomalies", markers=True)
anomaly_points = df[df["Anomaly"]]
fig.add_scatter(x=anomaly_points["Date"], y=anomaly_points["Sales"], mode='markers',
                marker=dict(color='red', size=10, symbol='x'), name='Anomaly')
fig.update_layout(height=250)

st.plotly_chart(fig, use_container_width=True)

# --- Display anomalies ---
st.markdown("**⚠️ Detected Anomalies**")
if anomaly_points.empty:
    st.success("No anomalies detected in the recent data.")
else:
    st.warning(f"{len(anomaly_points)} anomalies detected. Please review the affected dates and sales values below:")
    st.dataframe(anomaly_points[["Date", "Sales"]], use_container_width=True)
    st.markdown("""
    ### What To Do:
    - Investigate if these are due to promotions, technical issues, or external events.
    - Review the team or region responsible for these dates.
    - Adjust forecasts and reports accordingly.
    """)
