# app.py
# Streamlit dashboard for Customer Value Prediction & Segmentation

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

CLV_MODEL_PATH = Path("CustomerValueModel.keras")


@st.cache_data
def load_data():
    """
    Load customer-level dataset with segments.
    Prefer customers_with_segments.csv, fall back to final_df.csv.
    """
    if Path("customers_with_segments.csv").exists():
        df = pd.read_csv("customers_with_segments.csv")
    elif Path("final_df.csv").exists():
        df = pd.read_csv("final_df.csv")
    else:
        raise FileNotFoundError(
            "No customers_with_segments.csv or final_df.csv found in project root."
        )

    if "kmeans_cluster" not in df.columns:
        st.warning("kmeans_cluster column not found – clustering labels are missing.")
    if "gmm_cluster" not in df.columns:
        st.info("gmm_cluster column not found – GMM labels will be hidden.")

    return df


@st.cache_data
def compute_behavioral_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple behavioral score between 0 and 100
    using scaled RFM + behavioral features.
    """
    df = df.copy()

    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "morning",
        "afternoon",
        "evening",
        "night",
        "is_weekend",
    ]
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        return df

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[existing])

    # simple average score
    score = scaled.mean(axis=1)
    df["behavioral_score"] = (score * 100).round(1)
    return df


def sidebar_info(df: pd.DataFrame):
    st.sidebar.markdown("### Dataset Overview")
    st.sidebar.write(f"**Customers:** {df['CustomerID'].nunique():,}")
    if "kmeans_cluster" in df.columns:
        st.sidebar.write(f"**KMeans segments:** {df['kmeans_cluster'].nunique()}")
    if "gmm_cluster" in df.columns:
        st.sidebar.write(f"**GMM segments:** {df['gmm_cluster'].nunique()}")


# ──────────────────────────────────────────────────────────────────────────────
# Page 1 – Customer Lookup & Real-Time Behavioral Scoring
# ──────────────────────────────────────────────────────────────────────────────
def page_customer_lookup(df: pd.DataFrame):
    st.header("🔎 Customer Lookup & Behavioral Scoring")

    customer_ids = sorted(df["CustomerID"].unique())
    selected_id = st.selectbox("Select Customer ID", customer_ids)

    row = df[df["CustomerID"] == selected_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Info")
        st.write(f"**Customer ID:** {int(row['CustomerID'])}")
        if "Country" in df.columns:
            st.write(f"**Country:** {row.get('Country', 'N/A')}")
        if "kmeans_cluster" in df.columns:
            st.write(f"**KMeans Segment:** {int(row['kmeans_cluster'])}")
        if "gmm_cluster" in df.columns:
            st.write(f"**GMM Segment:** {int(row['gmm_cluster'])}")

    with col2:
        st.subheader("Behavioral Score")
        if "behavioral_score" in df.columns:
            st.metric(
                label="Behavioral Score (0–100)",
                value=row["behavioral_score"],
            )
        else:
            st.info("Behavioral score not computed.")

    st.subheader("RFM & Behavioral Features")
    cols_to_show = [
        c
        for c in [
            "Recency",
            "Frequency",
            "Monetary",
            "morning",
            "afternoon",
            "evening",
            "night",
            "is_weekend",
        ]
        if c in df.columns
    ]
    st.write(row[cols_to_show])


# ──────────────────────────────────────────────────────────────────────────────
# Page 2 – Segment Explorer with Filters
# ──────────────────────────────────────────────────────────────────────────────
def page_segment_explorer(df: pd.DataFrame):
    st.header("🧩 Segment Explorer")

    if "kmeans_cluster" not in df.columns:
        st.error("kmeans_cluster column is missing – cannot explore segments.")
        return

    col1, col2 = st.columns(2)
    with col1:
        segment = st.selectbox(
            "Select KMeans Segment",
            sorted(df["kmeans_cluster"].unique()),
            format_func=lambda x: f"Segment {x}",
        )
    with col2:
        country_filter = (
            st.selectbox(
                "Filter by Country (optional)",
                ["All"] + sorted(df["Country"].dropna().unique().tolist())
                if "Country" in df.columns
                else ["All"],
            )
            if "Country" in df.columns
            else "All"
        )

    seg_df = df[df["kmeans_cluster"] == segment].copy()
    if country_filter != "All" and "Country" in seg_df.columns:
        seg_df = seg_df[seg_df["Country"] == country_filter]

    st.subheader("Segment KPIs")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Customers", f"{seg_df['CustomerID'].nunique():,}")
    if "Monetary" in df.columns:
        col_b.metric("Avg Monetary", f"{seg_df['Monetary'].mean():.2f}")
    if "Frequency" in df.columns:
        col_c.metric("Avg Frequency", f"{seg_df['Frequency'].mean():.2f}")
    if "purchase_flag" in df.columns:
        col_d.metric(
            "Conversion Rate",
            f"{100 * seg_df['purchase_flag'].mean():.1f}%",
        )

    st.subheader("RFM Distribution in Selected Segment")
    rfm_cols = [c for c in ["Recency", "Frequency", "Monetary"] if c in seg_df.columns]
    if rfm_cols:
        fig = px.box(
            seg_df,
            y=rfm_cols,
            points="outliers",
            title="RFM Boxplots",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Size by Country")
    if "Country" in seg_df.columns:
        country_counts = seg_df["Country"].value_counts().reset_index()
        country_counts.columns = ["Country", "Customers"]
        fig2 = px.bar(
            country_counts,
            x="Country",
            y="Customers",
            title="Customers per Country in this Segment",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Page 3 – Campaign Performance Simulator with ROI
# ──────────────────────────────────────────────────────────────────────────────
def page_campaign_simulator(df: pd.DataFrame):
    st.header("📈 Campaign Performance Simulator & ROI")

    if "kmeans_cluster" not in df.columns or "Monetary" not in df.columns:
        st.error("Need kmeans_cluster and Monetary columns for simulation.")
        return

    segment = st.selectbox(
        "Target KMeans Segment",
        sorted(df["kmeans_cluster"].unique()),
        format_func=lambda x: f"Segment {x}",
    )

    seg_df = df[df["kmeans_cluster"] == segment]

    st.markdown("### Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_customers = col1.number_input(
            "Number of customers to target",
            min_value=100,
            max_value=int(seg_df["CustomerID"].nunique()),
            value=min(1000, int(seg_df["CustomerID"].nunique())),
            step=100,
        )
    with col2:
        uplift = col2.number_input(
            "Expected uplift in conversion (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5,
        )
    with col3:
        cost_per_customer = col3.number_input(
            "Campaign cost per customer",
            min_value=0.0,
            value=0.5,
            step=0.1,
        )

    base_conv = seg_df["purchase_flag"].mean() if "purchase_flag" in seg_df.columns else 0.1
    avg_order_value = seg_df["Monetary"].mean()

    st.markdown("### Results")

    # expected conversions after uplift
    uplift_factor = 1 + uplift / 100.0
    expected_conv_rate = base_conv * uplift_factor
    expected_conversions = target_customers * expected_conv_rate
    expected_revenue = expected_conversions * avg_order_value
    total_cost = target_customers * cost_per_customer
    roi = (expected_revenue - total_cost) / total_cost if total_cost > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base conversion rate", f"{100 * base_conv:.1f}%")
    c2.metric("Expected conversion rate", f"{100 * expected_conv_rate:.1f}%")
    c3.metric("Expected revenue", f"{expected_revenue:.2f}")
    c4.metric("ROI", f"{100 * roi:.1f}%")

    st.info(
        "This simulator uses historical conversion rate and average monetary value "
        "for the selected segment, then applies your uplift and cost assumptions."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page 4 – Customer Lifetime Value (CLV) Forecasting
# ──────────────────────────────────────────────────────────────────────────────
def page_clv_dashboard(df: pd.DataFrame):
    st.header("💰 Customer Lifetime Value (CLV) Dashboard")

    used_cols = [c for c in ["Recency", "Frequency", "Monetary"] if c in df.columns]
    if not used_cols:
        st.error("Need Recency, Frequency and Monetary columns for CLV.")
        return

    # try to load keras model if present
    clv_model = None
    if CLV_MODEL_PATH.exists():
        try:
            from tensorflow.keras.models import load_model

            clv_model = load_model(CLV_MODEL_PATH)
            st.success("Loaded CLV model from CustomerValueModel.keras")
        except Exception as e:
            st.warning(f"Could not load CLV model: {e}")

    features = df[used_cols].values

    if clv_model is not None:
        # model-based CLV prediction
        clv_raw = clv_model.predict(features)
        clv_scores = clv_raw.reshape(-1)
    else:
        # simple heuristic CLV: Frequency * Monetary with scaling
        clv_scores = (df["Frequency"] * df["Monetary"]).values

    scaler = MinMaxScaler()
    clv_norm = scaler.fit_transform(clv_scores.reshape(-1, 1)).reshape(-1)
    df["CLV_score"] = (clv_norm * 100).round(1)

    st.subheader("CLV Score Distribution")
    fig = px.histogram(
        df,
        x="CLV_score",
        nbins=30,
        title="Distribution of CLV Scores",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Customers by CLV")
    top_n = st.slider("Show top N customers", 10, 200, 50, step=10)
    top_df = df.sort_values("CLV_score", ascending=False).head(top_n)
    st.dataframe(top_df[["CustomerID", "CLV_score", "Monetary", "Frequency"]])

    st.info(
        "CLV_score is scaled between 0 and 100. "
        "If a trained model is available it is used, otherwise a heuristic based on Frequency × Monetary is applied."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page 5 – Export Marketing Lists
# ──────────────────────────────────────────────────────────────────────────────
def page_export_lists(df: pd.DataFrame):
    st.header("📤 Export Segment-Based Marketing Lists")

    if "kmeans_cluster" not in df.columns:
        st.error("kmeans_cluster column is required to export segments.")
        return

    segs = sorted(df["kmeans_cluster"].unique())
    selected_segs = st.multiselect(
        "Select segments to export",
        segs,
        default=segs,
        format_func=lambda x: f"Segment {x}",
    )

    min_behavior_score = 0.0
    if "behavioral_score" in df.columns:
        min_behavior_score = st.slider(
            "Minimum behavioral score (0–100)",
            0.0,
            100.0,
            30.0,
            step=5.0,
        )

    export_df = df[df["kmeans_cluster"].isin(selected_segs)].copy()
    if "behavioral_score" in export_df.columns:
        export_df = export_df[export_df["behavioral_score"] >= min_behavior_score]

    st.write(f"Selected customers: {export_df['CustomerID'].nunique():,}")

    cols = ["CustomerID", "kmeans_cluster", "gmm_cluster"] if "gmm_cluster" in df.columns else ["CustomerID", "kmeans_cluster"]
    if "behavioral_score" in df.columns:
        cols.append("behavioral_score")
    if "CLV_score" in df.columns:
        cols.append("CLV_score")
    if "Country" in df.columns:
        cols.append("Country")

    export_view = export_df[cols].drop_duplicates()
    st.dataframe(export_view)

    csv_bytes = export_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV marketing list",
        data=csv_bytes,
        file_name="marketing_list.csv",
        mime="text/csv",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Customer Value & Segmentation Dashboard",
        layout="wide",
    )

    st.title("Customer Value Prediction & Segmentation Dashboard")

    df = load_data()
    df = compute_behavioral_scores(df)
    sidebar_info(df)

    pages = {
        "Customer lookup": page_customer_lookup,
        "Segment explorer": page_segment_explorer,
        "Campaign simulator": page_campaign_simulator,
        "CLV dashboard": page_clv_dashboard,
        "Export marketing lists": page_export_lists,
    }

    page_name = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[page_name](df)


if __name__ == "__main__":
    main()
