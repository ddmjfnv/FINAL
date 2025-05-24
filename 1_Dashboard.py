import os
import numpy as np
import altair as alt
import pandas as pd
os.system("pip install sklearn")
os.system("pip install faker")
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import faker as Faker
from sklearn.ensemble import IsolationForest
from datetime import datetime
import streamlit as st



st.set_page_config(page_title="AI-Solutions Dashboard", layout="wide", page_icon="📊")


# Custom styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        
    }
    .stButton>button {
        width: 100%;
    }
   
    #MainMenu, header, footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.error("You must log in to access the dashboard.")
    st.stop()
col1, col2 = st.columns([6, 1])  # Adjust widths as needed

with col1:
    st.markdown("**<p style='font-size: 15px;'>📊 AI-Solutions Sales Dashboard</p>**", unsafe_allow_html=True)  # or any text you want

with col2:
    if st.button("🔓 Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.rerun()




# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_ai_solutions_web_server_logs.csv", parse_dates=['Date', 'Subscription_End_Date'])
    df['Hour'] = pd.to_datetime(df['Time'],  format='%H:%M:%S', errors='coerce').dt.hour.fillna(0).astype(int)
    df['Day'] = df['Date'].dt.day_name()
    df['Week'] = df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    return df

df = load_data()

tabs = st.tabs([
    "📊 Dashboard", "📈 Sales Performance", "💹KPIs", "👥 Customer Insights", "🧠 AI Recommendations",
    "📣 Marketing", "📋 Leads", "💰 Finance", "🚨 Anomalies", "📄 Reports", "🌐 Live Traffic"
])


# --- TAB 1: Dashboard Overview ---
with tabs[0]:
       

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    view_mode = st.sidebar.radio("View Mode", ["Sales Team", "Sales Person"])
    start_date, end_date = st.sidebar.date_input("Date Range", [df['Date'].min(), df['Date'].max()])
    country = st.sidebar.multiselect("Country", ["All"] + sorted(df['Country'].dropna().unique()))
    product = st.sidebar.multiselect("Product", ["All"] + sorted(df['Product_Service_Interest'].dropna().unique()))
    hour = st.sidebar.selectbox("Hour", ["All"] + sorted(df['Hour'].unique()))
    selected_group = st.sidebar.selectbox(
        "Sales Team" if view_mode == "Sales Team" else "Sales Person",
        ["All"] + sorted(df['Sales_Team'].unique() if view_mode == "Sales Team" else df['Sales_Person'].unique())
    )

    # --- Apply filters ---
    filtered_df = df[
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ]

    if hour != "All":
        filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]
    if selected_group != "All":
        col = 'Sales_Team' if view_mode == "Sales Team" else 'Sales_Person'
        filtered_df = filtered_df[filtered_df[col] == selected_group]




    # --- Revenue Gauge ---
    current_month = pd.Timestamp(datetime.now().replace(day=1))
    filtered_df_month = df[df['Date'] >= current_month]
    current_revenue = filtered_df_month['Deal_Value_USD'].sum()
    monthly_target = 1_000_000
    current_revenue = filtered_df['Deal_Value_USD'].sum()
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_revenue,
        number={'valueformat': ',.0f'},  # ✅ Comma-separated, no "M"
        delta={'reference': monthly_target},
        title={'text': "Monthly Revenue Progress"},
        gauge={
            'axis': {'range': [None, monthly_target]},
            'bar': {'color': "#f04a17"},
            'steps': [
                {'range': [0, 0.5 * monthly_target], 'color': "#f38c05"},
                {'range': [0.5 * monthly_target, 0.9 * monthly_target], 'color': "#f1f508"},
                {'range': [0.9 * monthly_target, monthly_target], 'color': "#07f01a"}
            ]
        }
    ))

    fig_gauge.update_layout(height=250)

    # --- Weekly Trend Line Chart ---
    demo_df = df[df['Action_Type'].str.contains("demo", case=False, na=False)]
    assistant_df = df[df['Product_Service_Interest'].str.contains("assistant", case=False, na=False)]
    demo_trend = demo_df.groupby('Week').size().rename("Demo Requests")
    assistant_trend = assistant_df.groupby('Week').size().rename("Assistant Requests")
    trend_df = pd.concat([demo_trend, assistant_trend], axis=1).fillna(0).reset_index()
    fig_line = px.line(trend_df, x='Week', y=['Demo Requests', 'Assistant Requests'], title="📈 Weekly Trend", markers=True)
    fig_line.update_layout(height=250)

    # --- Heatmap ---
    heatmap_data = df.pivot_table(index='Day', columns='Hour', values='Session_ID', aggfunc='count').fillna(0)
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(ordered_days)
    fig_heatmap, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heatmap_data, cmap="Blues", ax=ax)
    ax.set_title("User Activity by Hour and Day")

    # --- Top Performing Products ---
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True, height=240):
            top_products = filtered_df.groupby("Product_Service_Interest")["Deal_Value_USD"].sum().sort_values(ascending=False).head(10)
            fig_top = px.bar(
                top_products,
                x=top_products.values,
                y=top_products.index,
                orientation='h',
                color=top_products.values,
                color_continuous_scale='Inferno',
                title="💡 Top Performing Products"
            )
            fig_top.update_layout(height=230, margin=dict(l=20, r=20, t=60, b=20), xaxis_title="Revenue (USD)", yaxis_title="Product")
            st.plotly_chart(fig_top, use_container_width=True)

    with c2:
        with st.container(
                            height=240
                        ):
            st.plotly_chart(fig_gauge, use_container_width=True)
            

    group_col = "Sales_Team" if view_mode == "Sales Team" else "Sales_Person"

    # --- Side-by-Side Charts: Avg Deal Value + Top Products ---
    col3, col4 = st.columns(2)

    with col3:
        with st.container(height=240):
            if group_col in filtered_df.columns:
                avg_deal_df = filtered_df.groupby(group_col).apply(
                    lambda x: x["Deal_Value_USD"].sum() / x["Session_ID"].nunique()
                ).reset_index(name="Avg_Deal_Per_Session")
                avg_deal_df = avg_deal_df.sort_values("Avg_Deal_Per_Session", ascending=False)

                fig_avg = px.bar(
                    avg_deal_df, x=group_col, y="Avg_Deal_Per_Session",
                    title=f"💰 Avg Deal Value per Session by {group_col.replace('_', ' ')}",
                    color="Avg_Deal_Per_Session", color_continuous_scale="Blues"
                )
                
                st.plotly_chart(fig_avg, use_container_width=True)
            else:
                st.warning(f"{group_col} not found in dataset.")


    with col4:
        with st.container(
            height=240
        ):
            product_view_df = filtered_df.groupby([group_col, "Product_Service_Interest"]).size().reset_index(name="Views")
            top_viewed = product_view_df.sort_values("Views", ascending=False).head(20)
            fig_top_products = px.bar(
                top_viewed, x="Views", y="Product_Service_Interest",
                color=group_col, orientation="h",
                title=f"Top Viewed Products by {group_col.replace('_', ' ')}"
            )
            fig_top_products.update_layout(height= 230)
            st.plotly_chart(fig_top_products, use_container_width=True)


# --- TAB 2: Sales Performance ---
with tabs[1]:
    st.markdown("Explore sales forecasting, churn trends, profitability, and anomaly detection using advanced analytics."
            
                )

    c1, c2 = st.columns(2)
    # --- 1. Sales Forecasting ---

    with c1:
        with st.container(border=True, height=240):
            forecast_df = filtered_df.groupby("Date")["Deal_Value_USD"].sum().reset_index().sort_values("Date")
            forecast_df["Day_Num"] = (forecast_df["Date"] - forecast_df["Date"].min()).dt.days
            X = forecast_df[["Day_Num"]]
            y = forecast_df["Deal_Value_USD"]

            model = LinearRegression()
            model.fit(X, y)
            forecast_df["Prediction"] = model.predict(X)

            fig1 = px.line(forecast_df, x="Date", y=["Deal_Value_USD", "Prediction"],
                        labels={"value": "USD", "variable": "Type"},
                        title="📊 Sales Forecasting (Actual vs Predicted Deal Value Over Time)")
            fig1.update_layout(height=230)
            st.plotly_chart(fig1, use_container_width=True)

    # --- 2. Churn Rate Estimation ---

    # --- 3. Revenue and Profit Analysis ---
    with c2:
        with st.container(border=True, height=240):
            revenue_df = filtered_df.groupby("Date").agg({"Deal_Value_USD": "sum"}).reset_index()
            revenue_df["Estimated_Profit"] = revenue_df["Deal_Value_USD"] * 0.25  # Assume 25% margin

            fig2 = px.area(revenue_df, x="Date", y=["Deal_Value_USD", "Estimated_Profit"],
                        labels={"value": "USD", "variable": "Metric"},
                        title="💰 Revenue vs Estimated Profit Over Time")
            fig2.update_layout(height=230)
            st.plotly_chart(fig2, use_container_width=True)

    # --- 4. Anomaly Detection ---

    with st.container(border=True, height=230):
        anomaly_df = filtered_df.groupby("Date")["Deal_Value_USD"].sum().reset_index()
        anomaly_df["Day_Num"] = (anomaly_df["Date"] - anomaly_df["Date"].min()).dt.days
        clf = IsolationForest(contamination=0.1, random_state=42)
        anomaly_df["Anomaly"] = clf.fit_predict(anomaly_df[["Day_Num", "Deal_Value_USD"]])
        anomaly_df["Anomaly"] = anomaly_df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

        fig3 = px.scatter(anomaly_df, x="Date", y="Deal_Value_USD", color="Anomaly",
                        title="⚠️ Sales Anomaly Detection", color_discrete_map={"Normal": "blue", "Anomaly": "red"})
        fig3.update_layout(height=240)
        st.plotly_chart(fig3, use_container_width=True)


with tabs[2]:

    # --- Metric Config ---

    metrics_config = [
            ("Total Deal Value (USD)", "Deal_Value_USD", "#29b5e8", False),
            ("Product Interests", "Product_Service_Interest", "#FF9F36", True),
            ("Demo Requests", "Action_Type", "#D45B90", True, lambda x: x.str.contains("demo", case=False, na=False)),
            ("VA Interest", "Product_Service_Interest", "#7D44CF", True, lambda x: x.str.contains("assistant", case=False, na=False)),
            ("Job Requests", "Job_Category_Interest", "#4CAF50", True)
        ]
    
       # ---KPIs top ---
    st.markdown("###### 🧠 Summary Insights")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        with st.container(height=85):
            st.metric("👥 Unique Visitors", f"{filtered_df['Session_ID'].nunique():,}")
    with k2:
        with st.container(height=85):
            st.metric("💵 Revenue", f"${filtered_df['Deal_Value_USD'].sum():,.2f}")
    with k3:
        with st.container(height=85):
            st.metric("🏆 Top Product", filtered_df['Product_Service_Interest'].mode().iloc[0] if not filtered_df.empty else "N/A")
    with k4:
        with st.container(height=85):
            st.metric("🥇 Top Performer", filtered_df['Sales_Person'].mode().iloc[0] if not filtered_df.empty else "N/A")

    with k5:
        with st.container(height=85):    
            # Simulate churn by seeing how many customers' subscriptions ended and didn’t renew
            subscription_df = filtered_df.copy()
            subscription_df["Is_Churned"] = subscription_df["Subscription_End_Date"] < subscription_df["Date"].max()
            churn_rate = subscription_df["Is_Churned"].mean()
            st.metric(label="📉 Churn Rate Estimation", value=f"{churn_rate*100:.2f}%")



    # --- Metric Display ---
    st.markdown("###### 📈 Key Metrics")
    metric_cols = st.columns(len(metrics_config))

    for col, (label, column, color, is_categorical, *condition) in zip(metric_cols, metrics_config):
        temp_df = filtered_df.copy()
        if condition:
            temp_df = temp_df[condition[0](temp_df[column])]
        elif is_categorical:
            temp_df = temp_df[temp_df[column] != "N/A"]

        value = temp_df[column].sum() if not is_categorical else len(temp_df)

        with col:
            with st.container(border=True):
                st.metric(label, f"{value:,}")
                # Chart
                chart_df = temp_df.groupby(temp_df["Date"].dt.date).size().reset_index(name="Count")
                fig = go.Figure(go.Bar(x=chart_df["Date"], y=chart_df["Count"], marker_color=color))
                fig.update_layout(height=120, margin=dict(l=0, r=0, t=10, b=10), xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)

 
# --- TAB 3: Customer Insights ---
with tabs[3]:
    st.header(" 👥 Customer Insights")
    tab3A, tab3B = st.columns(2)
    with tab3A:    
        
        st.subheader("Job Categories of Interest")
        job_counts = df["Job_Category_Interest"].value_counts().head(10)
        st.bar_chart(job_counts)

    geo = df["Country"].value_counts().reset_index()
    geo_counts = geo["Country"].value_counts().reset_index()
    geo_counts.columns = ["Country", "User_Count"]
    
    with tab3B:
        st.subheader("Users by Country")   
        fig = px.pie(geo_counts, names="Country", values="User_Count", height=300)
        st.plotly_chart(fig, use_container_width=True)
        

# --- TAB 4: AI Recommendations ---
with tabs[4]:
    st.header("🧠 AI-Powered Recommendations")
    st.info("This section could host product recommendations using collaborative filtering or similarity scores.")

    top_interest = df["Product_Service_Interest"].value_counts().head(5)
    st.subheader("Popular Products")
    st.write(top_interest)

# --- TAB 5: Marketing Attribution ---
with tabs[5]:
    tab5a, tab5b = st.columns(2)
    
    with tab5a:
        st.header("📣 Marketing Attribution")
        st.bar_chart(df["Lead_Source"].value_counts())
    with tab5b:
        st.subheader("Resource Request Paths")
        st.write(df["Requested_Resource"].value_counts().head(10))

# --- TAB 6: Lead Management ---
with tabs[6]:
    st.header("📋 Lead Management")
    leads = df[df["Lead_Source"].notna()]
    st.dataframe(leads[["Date", "Lead_Source", "Sales_Person", "Country", "Deal_Value_USD"]].head(50))

# --- TAB 7: Finance & Profitability ---
with tabs[7]:
    st.subheader("💰 Revenue & Profitability")
    df["Profit_Estimate"] = df["Deal_Value_USD"] * 0.3  # Assume 30% margin
    st.metric("Total Estimated Profit", f"${df['Profit_Estimate'].sum():,.0f}")
    fig = px.line(df, x="Date", y="Profit_Estimate", title="Profit Over Time")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 8: Anomaly Detection ---
with tabs[8]:
    st.markdown("###### 🚨 Anomaly Detection")
    
    # Group daily sales
    daily = df.groupby("Date")["Deal_Value_USD"].sum().reset_index()
    daily["Z_Score"] = (daily["Deal_Value_USD"] - daily["Deal_Value_USD"].mean()) / daily["Deal_Value_USD"].std()

    # Detect anomalies (Z-score > 2 or < -2)
    anomalies = daily[np.abs(daily["Z_Score"]) > 2]
    
        
    # Line chart with anomaly markers
    fig = px.line(daily, x="Date", y="Deal_Value_USD", title="Daily Deal Value with Anomalies", markers=True)
    fig.add_scatter(
        x=anomalies["Date"], y=anomalies["Deal_Value_USD"],
        mode='markers',
        marker=dict(color='red', size=10, symbol='x'),
        name="Anomaly"
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Display warning and anomaly table

    if anomalies.empty:
        st.success("✅ No significant anomalies detected in daily sales.")
    else:
        st.warning(f"⚠️ Detected {len(anomalies)} potential sales anomalies.")
        st.dataframe(anomalies, use_container_width=True)
    st.markdown("""  
    **What to do:**  
    - Investigate dates with sharp spikes or drops  
    - Check for marketing campaigns, outages, or reporting errors  
    - Alert the sales or finance team if needed  
    """)


# --- TAB 9: Custom Reports ---
with tabs[9]:
    st.header("📄 Custom Reports Builder")
    cols = df.select_dtypes(include=[np.number, "object"]).columns
    selected_cols = st.multiselect("Select columns to include in the report", cols.tolist(), default=cols.tolist()[:5])
    st.dataframe(df[selected_cols])

# --- TAB 10: Live Web Traffic Monitor ---
with tabs[10]:
    st.header("🌐 Live Web Traffic Monitor")
    st.write("Real-time streaming not implemented, but here’s a simulation.")
    recent = df.sort_values("Date", ascending=False).head(10)
    st.table(recent[["Date", "Sales_Person", "Requested_Resource", "Country"]])
