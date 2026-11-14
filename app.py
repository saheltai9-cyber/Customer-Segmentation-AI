"""
Customer Intelligence AI — Premium SaaS Analytics & Segmentation Platform
Streamlit Application
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.ui_components import (
    inject_custom_css,
    render_sidebar_brand,
    render_sidebar_footer,
    render_page_header,
    render_kpi_card,
    derive_cluster_profile,
    ICONS,
    PRIMARY_NAVY,
    ACCENT_TEAL,
    BORDER_COLOR,
    CARD_BG,
    TEXT_MUTED
)
from src.visualization import (
    create_segmentation_scatter,
    create_age_vs_spending_scatter,
    create_cluster_donut,
    create_cluster_metrics_comparison,
    create_evaluation_charts,
    create_customer_benchmark_bars,
    create_customer_radar
)

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Intelligence AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom friendly SaaS design tokens & CSS
inject_custom_css()

# 2. Data & Model Loading (Cached)
DATA_PATH = os.path.join("data", "raw", "Mall_Customers.csv")
MODEL_PATH = os.path.join("models", "pipeline.joblib")
EVAL_PATH = os.path.join("reports", "cluster_evaluation.csv")

@st.cache_data
def get_cached_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset file missing at {DATA_PATH}. Please run train.py first.")
        st.stop()
    df = load_data(DATA_PATH)
    if "Genre" in df.columns:
        df = df.rename(columns={"Genre": "Gender"})
    return df

@st.cache_resource
def get_cached_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Trained model missing at {MODEL_PATH}. Please run train.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)

df_raw = get_cached_data()
pipeline = get_cached_pipeline()

# Compute cluster labels, stats, and personas ONCE (cached)
@st.cache_data
def compute_clustering(_pipeline, _df_raw):
    """Run scoring, aggregation, and persona derivation once and cache the results."""
    X_features = _df_raw[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = _pipeline.named_steps["scaler"]
    kmeans = _pipeline.named_steps["kmeans"]
    X_scaled = pd.DataFrame(scaler.transform(X_features), columns=X_features.columns)
    cluster_labels = kmeans.predict(X_scaled)
    df_scored = _df_raw.copy()
    df_scored["Cluster"] = cluster_labels

    overall_means = {
        "Age": df_scored["Age"].mean(),
        "Income": df_scored["Annual Income (k$)"].mean(),
        "Spending": df_scored["Spending Score (1-100)"].mean(),
    }

    cluster_agg = df_scored.groupby("Cluster").agg({
        "Age": "mean",
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean",
        "CustomerID": "count"
    }).rename(columns={"CustomerID": "Count"})

    personas = {}
    for c in cluster_agg.index:
        personas[c] = derive_cluster_profile(cluster_agg.loc[c], overall_means)

    return df_scored, overall_means, cluster_agg, personas

df_scored, overall_means, cluster_agg, personas = compute_clustering(pipeline, df_raw)

# 3. Sidebar Navigation
with st.sidebar:
    render_sidebar_brand()

    try:
        from streamlit_option_menu import option_menu
        nav_selection = option_menu(
            menu_title=None,
            options=["Overview", "Segmentation Explorer", "Cluster Insights", "Customer Prediction", "Data & Model"],
            icons=["grid-fill", "pie-chart-fill", "people-fill", "person-badge-fill", "database-fill"],
            default_index=0,
            styles={
                "container": {"padding": "0.2rem", "background-color": "#EEF2F6"},
                "icon": {"color": "#64748B", "font-size": "15px"},
                "nav-link": {
                    "font-size": "13.5px",
                    "font-weight": "500",
                    "text-align": "left",
                    "margin": "3px 0px",
                    "padding": "9px 12px",
                    "border-radius": "8px",
                    "color": "#334155",
                    "--hover-color": "#E7EDF3",
                },
                "nav-link-selected": {
                    "background-color": "#DCEFEB",
                    "color": "#0F766E",
                    "font-weight": "600",
                    "border-left": "3px solid #0F766E",
                    "border-radius": "6px",
                    "box-shadow": "none",
                },
            }
        )
    except Exception:
        nav_selection = st.radio(
            "Navigation",
            ["Overview", "Segmentation Explorer", "Cluster Insights", "Customer Prediction", "Data & Model"],
            label_visibility="collapsed"
        )

    render_sidebar_footer(k_clusters=len(cluster_agg))


# ==========================================
# SECTION 1: OVERVIEW
# ==========================================
if nav_selection == "Overview":
    render_page_header(
        title="Customer Intelligence Overview",
        subtitle="Understand customer behaviour and segment patterns through unsupervised machine learning.",
        badge="Live Production Pipeline"
    )

    # 4 Real KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Total Customers",
            value=f"{len(df_scored):,}",
            subtext="Complete verified cohort records",
            svg_icon=ICONS["users"]
        )
    with col2:
        render_kpi_card(
            title="Identified Segments",
            value=f"{len(cluster_agg)}",
            subtext="K-Means optimal cluster count",
            svg_icon=ICONS["layers"]
        )
    with col3:
        render_kpi_card(
            title="Avg Annual Income",
            value=f"${overall_means['Income']:.1f}k",
            subtext="Across all shopping demographics",
            svg_icon=ICONS["dollar"]
        )
    with col4:
        render_kpi_card(
            title="Avg Spending Score",
            value=f"{overall_means['Spending']:.1f}/100",
            subtext="Normalized behavioral metric",
            svg_icon=ICONS["trend"]
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Primary Visualizations Row inside Unified Card Containers
    chart_col1, chart_col2 = st.columns([1.6, 1.0])
    with chart_col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">Customer Segmentation Landscape</div>
                <div class="card-subtitle">Annual Income vs Spending Score with cluster centroids</div>
                """,
                unsafe_allow_html=True
            )
            fig_scatter = create_segmentation_scatter(df_scored, cluster_col="Cluster", personas=personas)
            st.plotly_chart(fig_scatter, use_container_width=True)

    with chart_col2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">Cohort Distribution</div>
                <div class="card-subtitle">Customer share by identified segment</div>
                """,
                unsafe_allow_html=True
            )
            fig_donut = create_cluster_donut(df_scored, cluster_col="Cluster", personas=personas)
            st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Cohort Summary Table inside Unified Card Container
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-title">Executive Cohort Summary</div>
            <div class="card-subtitle">Empirical breakdown of all customer segments derived from live KMeans clustering</div>
            """,
            unsafe_allow_html=True
        )

        table_data = []
        for c in sorted(cluster_agg.index):
            p = personas[c]
            row = cluster_agg.loc[c]
            pct = (row['Count'] / len(df_scored)) * 100
            table_data.append({
                "Cluster ID": f"Cluster {c}",
                "Segment Persona": p["persona"],
                "Archetype": p["archetype"],
                "Cohort Size": f"{int(row['Count'])} ({pct:.1f}%)",
                "Avg Age": f"{row['Age']:.1f} yrs",
                "Avg Income": f"${row['Annual Income (k$)']:.1f}k",
                "Avg Spending": f"{row['Spending Score (1-100)']:.1f} / 100",
                "Primary Action Strategy": p["strategy"][:60] + "..."
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ==========================================
# SECTION 2: SEGMENTATION EXPLORER
# ==========================================
elif nav_selection == "Segmentation Explorer":
    render_page_header(
        title="Segmentation Explorer",
        subtitle="Multi-dimensional behavioral and demographic distribution inspection.",
        badge="Interactive Analytics"
    )

    exp_col1, exp_col2 = st.columns([1, 1])

    with exp_col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">Income vs Spending Propensity</div>
                <div class="card-subtitle">Circular markers highlight cluster mathematical centroids</div>
                """,
                unsafe_allow_html=True
            )
            fig_inc_spend = create_segmentation_scatter(df_scored, cluster_col="Cluster", personas=personas)
            st.plotly_chart(fig_inc_spend, use_container_width=True)

    with exp_col2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">Age Demographics vs Spending Score</div>
                <div class="card-subtitle">Generational engagement patterns and spending velocity</div>
                """,
                unsafe_allow_html=True
            )
            fig_age_spend = create_age_vs_spending_scatter(df_scored, cluster_col="Cluster", personas=personas)
            st.plotly_chart(fig_age_spend, use_container_width=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Metric Comparison across all clusters inside Unified Card Container
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-title">Cross-Cluster Attribute Benchmarking</div>
            <div class="card-subtitle">Side-by-side comparison of average age, income, and spending score across segments</div>
            """,
            unsafe_allow_html=True
        )
        fig_comparison = create_cluster_metrics_comparison(cluster_agg, personas=personas)
        st.plotly_chart(fig_comparison, use_container_width=True)


# ==========================================
# SECTION 3: CLUSTER INSIGHTS
# ==========================================
elif nav_selection == "Cluster Insights":
    render_page_header(
        title="Cluster Insights & Actionable Personas",
        subtitle="Data-derived commercial archetypes with real cohort characteristics and strategic playbooks.",
        badge="Behavioral Intelligence"
    )

    # 3-column responsive grid (2 rows of 3 cards for 6 clusters)
    c_list = sorted(cluster_agg.index)
    cols = st.columns(3)

    for idx, c in enumerate(c_list):
        p = personas[c]
        row = cluster_agg.loc[c]
        pct = (row["Count"] / len(df_scored)) * 100

        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {p['tag_color']}; padding-left: 0.75rem; margin-bottom: 0.85rem;">
                        <span class="badge-pill" style="background-color: {p.get('bg_pastel', '#F0FDFA')}; color: {p['tag_color']}; border-color: {p['tag_color']}30; font-size: 0.72rem;">
                            CLUSTER 0{c} &bull; {int(row['Count'])} customers ({pct:.1f}%)
                        </span>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {PRIMARY_NAVY}; margin-top: 0.35rem; line-height: 1.25;">
                            {p['persona']}
                        </div>
                        <div style="font-size: 0.78rem; color: {TEXT_MUTED}; font-weight: 500;">
                            {p['archetype']}
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; background: #F8FAFC; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.65rem 0.75rem; margin-bottom: 0.85rem; text-align: center;">
                        <div>
                            <div style="font-size: 0.7rem; color: {TEXT_MUTED}; font-weight: 600; text-transform: uppercase;">Age</div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: {PRIMARY_NAVY};">{row['Age']:.1f}y</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: {TEXT_MUTED}; font-weight: 600; text-transform: uppercase;">Income</div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: {PRIMARY_NAVY};">${row['Annual Income (k$)']:.0f}k</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: {TEXT_MUTED}; font-weight: 600; text-transform: uppercase;">Spend</div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: {PRIMARY_NAVY};">{row['Spending Score (1-100)']:.0f}</div>
                        </div>
                    </div>

                    <div style="font-size: 0.84rem; color: #475569; line-height: 1.5; margin-bottom: 0.75rem; min-height: 55px;">
                        {p['description']}
                    </div>

                    <div style="background: {p.get('bg_pastel', '#F8FAFC')}; border: 1px solid {p['tag_color']}25; border-radius: 8px; padding: 0.65rem 0.8rem; font-size: 0.82rem; color: {PRIMARY_NAVY}; line-height: 1.45;">
                        <span style="font-weight: 700; color: {p['tag_color']};">Strategy:</span> {p['strategy']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)


# ==========================================
# SECTION 4: CUSTOMER PREDICTION
# ==========================================
elif nav_selection == "Customer Prediction":
    render_page_header(
        title="Customer Segment Classifier",
        subtitle="Score a customer profile through the production pipeline to assign cluster and extract tailored playbooks.",
        badge="Inference Pipeline"
    )

    pred_col1, pred_col2 = st.columns([1.0, 1.3])

    with pred_col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">Customer Profile Inputs</div>
                <div class="card-subtitle">Adjust demographic and behavioral attributes</div>
                """,
                unsafe_allow_html=True
            )

            with st.form("customer_prediction_form"):
                in_age = st.slider("Customer Age (Years)", min_value=18, max_value=75, value=32, step=1)
                in_income = st.slider("Annual Income (k$)", min_value=15, max_value=140, value=75, step=1)
                in_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=78, step=1)

                st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Classify Customer Segment")

    with pred_col2:
        # Default or calculated prediction
        cust_input_df = pd.DataFrame([[in_age, in_income, in_score]], columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
        scaled_input = pd.DataFrame(pipeline.named_steps["scaler"].transform(cust_input_df), columns=cust_input_df.columns)
        pred_cluster = int(pipeline.named_steps["kmeans"].predict(scaled_input)[0])

        pred_persona = personas[pred_cluster]
        clust_stats = cluster_agg.loc[pred_cluster]

        if submitted:
            st.toast(f"Assigned to {pred_persona['persona']} (Cluster {pred_cluster})")

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="border-left: 5px solid {pred_persona['tag_color']}; padding-left: 0.85rem; margin-bottom: 0.9rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.35rem;">
                        <span class="badge-pill" style="background-color: {pred_persona.get('bg_pastel', '#F0FDFA')}; color: {pred_persona['tag_color']}; border-color: {pred_persona['tag_color']}30;">
                            Assigned: Cluster {pred_cluster} &bull; {pred_persona['archetype']}
                        </span>
                        <span style="font-size: 0.75rem; color: {TEXT_MUTED}; font-weight: 500;">Pipeline Classification</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: {PRIMARY_NAVY}; margin-top: 0.2rem;">
                        {pred_persona['persona']}
                    </div>
                </div>

                <div style="font-size: 0.88rem; color: #475569; margin-bottom: 0.9rem; line-height: 1.5;">
                    {pred_persona['description']}
                </div>

                <div style="background: {pred_persona.get('bg_pastel', '#F8FAFC')}; border: 1px solid {pred_persona['tag_color']}25; border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: {pred_persona['tag_color']}; text-transform: uppercase; margin-bottom: 0.3rem;">
                        Recommended Engagement Strategy
                    </div>
                    <div style="font-size: 0.88rem; color: {PRIMARY_NAVY}; line-height: 1.5;">
                        {pred_persona['strategy']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="card-title" style="font-size: 0.95rem;">Customer Attributes vs Cluster Benchmark</div>
                <div class="card-subtitle" style="margin-bottom: 0.5rem;">Direct comparison against Cluster {pred_cluster} averages</div>
                """,
                unsafe_allow_html=True
            )
            benchmark_fig = create_customer_benchmark_bars(
                customer_data={"Age": in_age, "Annual Income (k$)": in_income, "Spending Score (1-100)": in_score},
                cluster_stats=clust_stats
            )
            st.plotly_chart(benchmark_fig, use_container_width=True)


# ==========================================
# SECTION 5: DATA & MODEL
# ==========================================
elif nav_selection == "Data & Model":
    render_page_header(
        title="Dataset Architecture & Model Evaluation",
        subtitle="Transparent evaluation diagnostics, cluster validation curves, and verified cohort records.",
        badge="Model Governance"
    )

    # Evaluation Section
    if os.path.exists(EVAL_PATH):
        eval_df = pd.read_csv(EVAL_PATH)
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">Model Selection Diagnostics (k = 2 to 10)</div>
                <div class="card-subtitle">Elbow curve (Inertia SSE) and validation curves (Silhouette & Davies-Bouldin)</div>
                """,
                unsafe_allow_html=True
            )
            fig_eval = create_evaluation_charts(eval_df, selected_k=6)
            st.plotly_chart(fig_eval, use_container_width=True)

            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 1rem 1.15rem; margin-top: 0.75rem;">
                    <div style="font-size: 0.88rem; font-weight: 700; color: {PRIMARY_NAVY}; margin-bottom: 0.35rem;">
                        Optimal Cluster Count Justification: k = 6
                    </div>
                    <p style="font-size: 0.84rem; color: #475569; line-height: 1.55; margin: 0;">
                        Across candidate cluster counts k &isin; [2, 10], <strong>k = 6</strong> delivers the maximum 
                        <strong>Silhouette Score of 0.4311</strong> and a minimised <strong>Davies-Bouldin Index of 0.8350</strong>. 
                        The Elbow curve demonstrates an observable inflection point (diminishing returns in within-cluster sum of squares) at k=6. 
                        Commercially, 6 segments resolve distinct demographic quadrants (High Income/High Spend, High Income/Low Spend, Low Income/High Spend, Low Income/Low Spend, Young Professionals, and Mature Shoppers) with high actionable value and zero artificial over-segmentation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Pipeline Specifications Card
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="card-title">Scikit-Learn Production Pipeline Specification</div>
            <div class="card-subtitle">End-to-end preprocessing standard scaler and KMeans clustering engine</div>
            <div style="font-family: monospace; font-size: 0.82rem; background: #F8FAFC; border: 1px solid {BORDER_COLOR}; padding: 0.85rem 1rem; border-radius: 8px; line-height: 1.6;">
                <strong>Step 1:</strong> StandardScaler(with_mean=True, with_std=True) on [Age, Annual Income (k$), Spending Score (1-100)]<br>
                <strong>Step 2:</strong> KMeans(n_clusters=6, init='k-means++', random_state=42, n_init='auto')<br>
                <strong>Persistence:</strong> <code>models/pipeline.joblib</code> &bull; <strong>Status:</strong> Ready &amp; Active
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Dataset Explorer
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-title">Verified Customer Dataset Records (N = 200)</div>
            <div class="card-subtitle">Full cohort records with assigned cluster classification</div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(df_scored, use_container_width=True)
