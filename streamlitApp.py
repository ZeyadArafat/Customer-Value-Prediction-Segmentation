import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

PX_TEMPLATE = "plotly_white" 


def style_dataframe(df: pd.DataFrame, precision: int = 2):
    return (
        df.style
        .format(precision=precision)
        .set_properties(
            **{
                "color": "black",
                "font-size": "15px",
                "border-color": "#E0E0E0",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-size", "16px"),
                        ("color", "black"),
                        ("font-weight", "600"),
                        ("background-color", "#F5F5F5"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("color", "black"),
                        ("background-color", "white"),
                    ],
                },
            ]
        )
    )


def tidy_fig(fig, title: str = None):
    if title is not None:
        fig.update_layout(title=title)
    fig.update_layout(
        template=PX_TEMPLATE,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(size=13, color="black"),
    )
    return fig


# ───────────────────────────────────────────────
# Load Data
# ───────────────────────────────────────────────
@st.cache_data
def load_country_mapping():
    """
    Build a mapping CustomerID -> Country name
    from Online Retail.xlsx (most frequent country per customer).
    """
    excel_path = Path("Online Retail.xlsx")
    if not excel_path.exists():
        return None

    raw = pd.read_excel(excel_path, usecols=["CustomerID", "Country"])
    raw = raw.dropna(subset=["CustomerID", "Country"])

    raw["CustomerID"] = raw["CustomerID"].astype(int)

    mapping = (
        raw.groupby("CustomerID")["Country"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
    )

    return mapping


@st.cache_data
def load_data():
    if Path("customers_with_segments.csv").exists():
        df = pd.read_csv("customers_with_segments.csv")
    elif Path("final_df.csv").exists():
        df = pd.read_csv("final_df.csv")
    else:
        raise FileNotFoundError(
            "No customers_with_segments.csv or final_df.csv found in project root."
        )

    if "CustomerID" in df.columns:
        country_map = load_country_mapping()
        if country_map is not None:
            df = df.copy()
            df["CustomerID"] = df["CustomerID"].astype(int)
            df["Country_name"] = df["CustomerID"].map(country_map)
            df["Country"] = df["Country_name"].fillna(df.get("Country"))
            df.drop(columns=["Country_name"], inplace=True)

    return df


# ───────────────────────────────────────────────
# Compute Behavioral Score + CLV Score
# ───────────────────────────────────────────────
@st.cache_data
def compute_behavioral_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # behavioral_score
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
    if existing:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[existing])
        df["behavioral_score"] = (scaled.mean(axis=1) * 100).round(1)

    # CLV_score = scaled(Frequency × Monetary)
    if all(c in df.columns for c in ["Frequency", "Monetary"]):
        raw_clv = df["Frequency"] * df["Monetary"]
        clv_scaler = MinMaxScaler()
        clv_norm = clv_scaler.fit_transform(raw_clv.values.reshape(-1, 1)).reshape(-1)
        df["CLV_score"] = (clv_norm * 100).round(1)

    return df


# ───────────────────────────────────────────────
# Sidebar Info
# ───────────────────────────────────────────────
def sidebar_info(df):
    st.sidebar.markdown("### Dataset Overview")
    st.sidebar.write(f"**Customers:** {df['CustomerID'].nunique():,}")
    if "kmeans_cluster" in df.columns:
        st.sidebar.write(f"**KMeans segments:** {df['kmeans_cluster'].nunique()}")
    if "gmm_cluster" in df.columns:
        st.sidebar.write(f"**GMM segments:** {df['gmm_cluster'].nunique()}")


# ───────────────────────────────────────────────
# PAGE 1 – Customer Lookup
# ───────────────────────────────────────────────
def page_customer_lookup(df):

    st.header("🔎 Customer Lookup & Behavioral Scoring")

    selected_id = st.selectbox("Select Customer ID", sorted(df["CustomerID"].unique()))
    row = df[df["CustomerID"] == selected_id].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Info")
        st.write(f"**Customer ID:** {int(row['CustomerID'])}")
        if "Country" in df.columns:
            st.write(f"**Country:** {row['Country']}")
        if "kmeans_cluster" in df.columns:
            st.write(f"**KMeans Segment:** {int(row['kmeans_cluster'])}")
        if "gmm_cluster" in df.columns:
            st.write(f"**GMM Segment:** {int(row['gmm_cluster'])}")

    with col2:
        st.subheader("Behavioral Score")
        if "behavioral_score" in df.columns:
            st.metric("Behavioral Score (0–100)", row["behavioral_score"])

    st.subheader("RFM & Behavioral Features")
    cols = [
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

    rfm_df = row[cols].to_frame(name="Value")
    st.dataframe(
        style_dataframe(rfm_df, precision=3),
        use_container_width=True,
    )


# ───────────────────────────────────────────────
# PAGE 2 – Segment Explorer
# ───────────────────────────────────────────────
def page_segment_explorer(df):

    st.header("🧩 Segment Explorer")

    if "kmeans_cluster" not in df.columns:
        st.error("Missing kmeans_cluster")
        return

    col1, col2 = st.columns(2)

    with col1:
        segment = st.selectbox(
            "Select KMeans Segment",
            sorted(df["kmeans_cluster"].unique()),
            format_func=lambda x: f"Segment {x}",
        )

    with col2:
        if "Country" in df.columns:
            country_filter = st.selectbox(
                "Filter by Country",
                ["All"] + sorted(df["Country"].dropna().unique().tolist()),
            )
        else:
            country_filter = "All"

    seg_df = df[df["kmeans_cluster"] == segment]
    if country_filter != "All" and "Country" in seg_df.columns:
        seg_df = seg_df[seg_df["Country"] == country_filter]

    st.subheader("Segment KPIs")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", f"{seg_df['CustomerID'].nunique():,}")
    if "Monetary" in seg_df.columns:
        c2.metric("Avg Monetary", f"{seg_df['Monetary'].mean():.2f}")
    if "Frequency" in seg_df.columns:
        c3.metric("Avg Frequency", f"{seg_df['Frequency'].mean():.2f}")
    if "purchase_flag" in seg_df.columns:
        c4.metric("Conversion Rate", f"{seg_df['purchase_flag'].mean()*100:.1f}%")

    st.subheader("RFM Distribution")
    rfm_cols = [c for c in ["Recency", "Frequency", "Monetary"] if c in seg_df.columns]
    if rfm_cols:
        fig = px.box(seg_df, y=rfm_cols, points="outliers")
        tidy_fig(fig, "RFM Boxplots")
        st.plotly_chart(fig, use_container_width=True)

    if "Country" in seg_df.columns:
        st.subheader("Segment Size by Country")
        counts = seg_df["Country"].value_counts().reset_index()
        counts.columns = ["Country", "Customers"]
        fig2 = px.bar(counts, x="Country", y="Customers")
        tidy_fig(fig2, "Customers per Country in this Segment")
        st.plotly_chart(fig2, use_container_width=True)


# ───────────────────────────────────────────────
# PAGE 3 – Campaign Simulator
# ───────────────────────────────────────────────
def page_campaign_simulator(df):

    st.header("📈 Campaign Performance Simulator & ROI")

    if "kmeans_cluster" not in df.columns:
        st.error("Missing kmeans_cluster")
        return

    segment = st.selectbox(
        "Target Segment", sorted(df["kmeans_cluster"].unique())
    )

    seg_df = df[df["kmeans_cluster"] == segment]

    st.markdown("### Assumptions")
    c1, c2, c3 = st.columns(3)

    target_customers = c1.number_input(
        "Target customers",
        100,
        int(seg_df["CustomerID"].nunique()),
        min(1000, int(seg_df["CustomerID"].nunique())),
        100,
    )
    uplift = c2.number_input("Conversion uplift (%)", 0.0, 100.0, 5.0)
    cost = c3.number_input("Cost per customer", 0.0, 100.0, 0.5)

    base_conv = seg_df["purchase_flag"].mean() if "purchase_flag" in seg_df.columns else 0.1
    avg_order = seg_df["Monetary"].mean() if "Monetary" in seg_df.columns else 1.0

    uplift_factor = 1 + uplift / 100.0
    expected_conv_rate = base_conv * uplift_factor
    conversions = target_customers * expected_conv_rate
    revenue = conversions * avg_order
    total_cost = target_customers * cost
    roi = (revenue - total_cost) / total_cost if total_cost else 0

    st.subheader("Results")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Base Conversion", f"{base_conv*100:.1f}%")
    colB.metric("Expected Conversion", f"{expected_conv_rate*100:.1f}%")
    colC.metric("Expected Revenue", f"{revenue:.2f}")
    colD.metric("ROI", f"{roi*100:.1f}%")


# ───────────────────────────────────────────────
# PAGE 4 – CLV Dashboard (Heuristic)
# ───────────────────────────────────────────────
def page_clv_dashboard(df):

    st.header("💰 Customer Lifetime Value (CLV) Dashboard")

    if "CLV_score" not in df.columns:
        if not all(c in df.columns for c in ["Frequency", "Monetary"]):
            st.error("Missing Frequency/Monetary for CLV.")
            return
        raw_clv = df["Frequency"] * df["Monetary"]
        scaler = MinMaxScaler()
        df["CLV_score"] = (
            scaler.fit_transform(raw_clv.values.reshape(-1, 1)).reshape(-1) * 100
        ).round(1)

    fig = px.histogram(df, x="CLV_score", nbins=30)
    tidy_fig(fig, "Distribution of CLV Scores")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Customers by CLV")
    top_n = st.slider("Top N", 10, 200, 50)
    cols = ["CustomerID", "CLV_score", "Monetary", "Frequency"]
    if "Country" in df.columns:
        cols.append("Country")

    top_df = df.sort_values("CLV_score", ascending=False).head(top_n)[cols]
    st.dataframe(
        style_dataframe(top_df),
        use_container_width=True,
    )


# ───────────────────────────────────────────────
# PAGE 5 – Export Marketing Lists
# ───────────────────────────────────────────────
def page_export_lists(df):

    st.header("📤 Export Marketing Lists")

    if "kmeans_cluster" not in df.columns:
        st.error("kmeans_cluster is required.")
        return

    segs = sorted(df["kmeans_cluster"].unique())
    selected = st.multiselect("Select Segments", segs, default=segs)

    if "behavioral_score" in df.columns:
        min_score = st.slider("Min Behavioral Score", 0.0, 100.0, 30.0)
        data = df[df["kmeans_cluster"].isin(selected)]
        data = data[data["behavioral_score"] >= min_score]
    else:
        st.info("behavioral_score not found – exporting all selected customers.")
        data = df[df["kmeans_cluster"].isin(selected)]

    export_cols = ["CustomerID", "kmeans_cluster"]

    if "Country" in df.columns:
        export_cols.append("Country")
    if "behavioral_score" in df.columns:
        export_cols.append("behavioral_score")
    if "CLV_score" in df.columns:
        export_cols.append("CLV_score")

    out = data[export_cols].drop_duplicates()
    st.dataframe(
        style_dataframe(out),
        use_container_width=True,
    )

    st.download_button(
        "Download CSV",
        out.to_csv(index=False),
        "marketing_list.csv",
        "text/csv",
    )


# ───────────────────────────────────────────────
# MAIN APP
# ───────────────────────────────────────────────
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

    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[selection](df)


if __name__ == "__main__":
    main()
