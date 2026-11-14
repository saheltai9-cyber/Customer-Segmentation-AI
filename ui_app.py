"""
Customer Intelligence Studio — Premium NiceGUI Application
A professionally designed Customer Segmentation & Analytics SaaS platform.

Uses the existing trained K-Means pipeline from models/pipeline.joblib.
Does NOT retrain or duplicate ML logic.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from nicegui import ui, app

# ---------------------------------------------------------------------------
# ML BACKEND — reuse existing modules
# ---------------------------------------------------------------------------
from src.data_loader import load_data
from src.preprocessing import FEATURE_COLUMNS

DATA_PATH   = os.path.join("data", "raw", "Mall_Customers.csv")
MODEL_PATH  = os.path.join("models", "pipeline.joblib")
EVAL_PATH   = os.path.join("reports", "cluster_evaluation.csv")

# Load dataset once at startup
_df_raw = load_data(DATA_PATH)
if "Genre" in _df_raw.columns:
    _df_raw = _df_raw.rename(columns={"Genre": "Gender"})

_pipeline = joblib.load(MODEL_PATH)
_scaler   = _pipeline.named_steps["scaler"]
_kmeans   = _pipeline.named_steps["kmeans"]

X_feat = _df_raw[FEATURE_COLUMNS].copy()
X_scaled = pd.DataFrame(_scaler.transform(X_feat), columns=FEATURE_COLUMNS)
_df_raw["Cluster"] = _kmeans.predict(X_scaled)

_cluster_agg = _df_raw.groupby("Cluster").agg({
    "Age": "mean",
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean",
    "CustomerID": "count",
}).rename(columns={"CustomerID": "Count"})

_overall = {
    "total":    len(_df_raw),
    "clusters": len(_cluster_agg),
    "income":   _df_raw["Annual Income (k$)"].mean(),
    "spending": _df_raw["Spending Score (1-100)"].mean(),
    "age":      _df_raw["Age"].mean(),
}

_eval_df = pd.read_csv(EVAL_PATH) if os.path.exists(EVAL_PATH) else None

# ---------------------------------------------------------------------------
# DYNAMIC PERSONA GENERATION  (data-derived, never hardcoded)
# ---------------------------------------------------------------------------
def _persona(row):
    age, inc, sp = row["Age"], row["Annual Income (k$)"], row["Spending Score (1-100)"]
    il = "High" if inc >= 70 else ("Low" if inc < 40 else "Moderate")
    sl = "High" if sp >= 65  else ("Low" if sp < 35  else "Moderate")
    ab = "Young" if age < 30 else ("Mature" if age >= 50 else "Mid-Career")

    if il == "High" and sl == "High":
        return "Affluent High-Spenders", "#0D9488", "High-income customers with strong spending velocity. Target with VIP loyalty, exclusive previews, and premium concierge services."
    if il == "High" and sl == "Low":
        return "Affluent Conservatives", "#172B4D", "High-income but cautious spenders. Engage with value-driven premium messaging, investment-grade products, and wealth-aligned promotions."
    if il == "Low" and sl == "High":
        return "Aspirational Trendsetters", "#F59E0B", "Budget-conscious yet enthusiastic spenders. Offer flexible financing, flash sales, and trend-forward affordable lines."
    if il == "Low" and sl == "Low":
        return "Frugal Budget Shoppers", "#64748B", "Price-sensitive minimal spenders. Attract with essentials bundles, deep discounts, and clearance campaigns."
    if ab == "Mature":
        return "Steady Mature Consumers", "#2563EB", "Established older demographic with balanced habits. Provide dependable service, loyalty tenure perks, and traditional engagement."
    if ab == "Young":
        return "Emerging Professionals", "#7C3AED", "Early-career customers with moderate, growing spending. Use digital-first programs, referral bonuses, and lifestyle bundles."
    return f"{il}-Income {sl}-Spend Segment", "#0D9488", f"Average age {age:.0f}, earning ${inc:.0f}k with spending index {sp:.0f}/100."

_personas = {}
for c in _cluster_agg.index:
    name, color, desc = _persona(_cluster_agg.loc[c])
    _personas[c] = {"name": name, "color": color, "desc": desc, "row": _cluster_agg.loc[c]}

# ---------------------------------------------------------------------------
# COLOUR PALETTE  (cluster chart colours)
# ---------------------------------------------------------------------------
CLUSTER_COLORS = ["#0D9488", "#172B4D", "#2563EB", "#F59E0B", "#7C3AED", "#E11D48", "#059669", "#4F46E5"]

# ---------------------------------------------------------------------------
# GLOBAL CSS INJECTION
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root {
    --navy: #172B4D;
    --teal: #0F766E;
    --bg: #F5F7FA;
    --sidebar-bg: #EEF2F6;
    --sidebar-text: #334155;
    --active-nav-bg: #DCEFEB;
    --active-nav-text: #0F766E;
    --hover-bg: #E7EDF3;
    --card: #FFFFFF;
    --text: #172B4D;
    --muted: #64748B;
    --border: #E2E8F0;
    --teal-light: #DCEFEB;
}
body, .q-page, .nicegui-content { background: var(--bg) !important; font-family: 'Inter', sans-serif; color: var(--text); }
.q-drawer {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}
.q-drawer .q-item {
    color: var(--sidebar-text) !important;
    border-radius: 8px;
    margin: 3px 10px;
    padding: 8px 14px;
    font-size: 13.5px;
    font-weight: 500;
    transition: all .15s ease-in-out;
}
.q-drawer .q-item:hover {
    background: var(--hover-bg) !important;
    color: var(--navy) !important;
}
.q-drawer .q-item--active, .q-drawer .q-item.active-nav {
    background: var(--active-nav-bg) !important;
    color: var(--active-nav-text) !important;
    font-weight: 600 !important;
    border-left: 3px solid var(--active-nav-text) !important;
    border-radius: 6px;
}
.q-drawer .q-item__section--avatar, .q-drawer .q-icon {
    color: var(--muted) !important;
}
.q-drawer .q-item.active-nav .q-item__section--avatar,
.q-drawer .q-item.active-nav .q-icon,
.q-drawer .q-item.active-nav .material-icons {
    color: var(--active-nav-text) !important;
}
.kpi-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 22px 26px; box-shadow: 0 1px 3px rgba(16,24,40,.04); transition: transform .15s, box-shadow .15s; }
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(16,24,40,.07); }
.kpi-label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-bottom: 6px; }
.kpi-value { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 28px; font-weight: 700; color: var(--navy); letter-spacing: -.02em; line-height: 1.15; }
.kpi-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
.section-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 28px; box-shadow: 0 1px 3px rgba(16,24,40,.04); }
.page-title { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 26px; font-weight: 700; color: var(--navy); letter-spacing: -.02em; margin: 0; }
.page-sub { font-size: 14px; color: var(--muted); margin: 4px 0 0 0; }
.cluster-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 24px; box-shadow: 0 1px 3px rgba(16,24,40,.04); position: relative; overflow: hidden; }
.cluster-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; }
.badge { display: inline-block; padding: 3px 10px; font-size: 11px; font-weight: 600; border-radius: 20px; background: var(--teal-light); color: var(--teal); border: 1px solid rgba(15,118,110,.2); }
.topbar-wrap { background: var(--bg); border-bottom: 1px solid var(--border); padding: 14px 28px; display: flex; align-items: center; justify-content: space-between; }
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; background: #F8FAFC; border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; }
.stat-label { font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--muted); }
.stat-val { font-size: 18px; font-weight: 700; color: var(--navy); }
.pred-result { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 28px; border-top: 4px solid var(--teal); }
"""

# ===================================================================
# PAGE BUILDERS
# ===================================================================

def _build_overview(container):
    container.clear()
    with container:
        # Page header
        ui.html('<h1 class="page-title">Customer Intelligence Overview</h1>')
        ui.html('<p class="page-sub">Understand customer behaviour through unsupervised machine learning clustering.</p>')
        ui.separator().style("margin: 18px 0 24px 0; opacity: .25;")

        # KPI row
        with ui.row().style("width:100%; gap:20px; flex-wrap:wrap;"):
            _kpi("Total Customers", f"{_overall['total']:,}", "Verified cohort records", "groups")
            _kpi("Segments Found",  str(_overall['clusters']),  "K-Means optimal clusters", "hub")
            _kpi("Avg Annual Income", f"${_overall['income']:.1f}k", "Across all demographics", "payments")
            _kpi("Avg Spending Score", f"{_overall['spending']:.1f}/100", "Normalized engagement metric", "trending_up")

        ui.separator().style("margin: 28px 0; opacity: .12;")

        # Charts row
        with ui.row().style("width:100%; gap:20px; flex-wrap:wrap; align-items:flex-start;"):
            # Scatter chart
            with ui.column().style("flex:1.6; min-width:400px;"):
                ui.html('<div class="section-card">')
                ui.html('<div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:14px;">Customer Segmentation Landscape</div>')
                _scatter_chart()
                ui.html('</div>')

            # Donut chart
            with ui.column().style("flex:1; min-width:300px;"):
                ui.html('<div class="section-card">')
                ui.html('<div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:14px;">Cohort Distribution</div>')
                _donut_chart()
                ui.html('</div>')


def _build_segmentation(container):
    container.clear()
    with container:
        ui.html('<h1 class="page-title">Segmentation Explorer</h1>')
        ui.html('<p class="page-sub">Multi-dimensional behavioral and demographic distribution analysis.</p>')
        ui.separator().style("margin: 18px 0 24px 0; opacity: .25;")

        with ui.row().style("width:100%; gap:20px; flex-wrap:wrap;"):
            with ui.column().style("flex:1; min-width:400px;"):
                ui.html('<div class="section-card">')
                ui.html('<div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:6px;">Income vs Spending Score</div>')
                ui.html('<div style="font-size:12px;color:#697586;margin-bottom:14px;">Diamond markers = cluster centroids</div>')
                _scatter_chart()
                ui.html('</div>')

            with ui.column().style("flex:1; min-width:400px;"):
                ui.html('<div class="section-card">')
                ui.html('<div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:6px;">Age vs Spending Score</div>')
                ui.html('<div style="font-size:12px;color:#697586;margin-bottom:14px;">Generational engagement patterns</div>')
                _age_scatter_chart()
                ui.html('</div>')

        ui.separator().style("margin: 18px 0; opacity: .12;")

        ui.html('<div class="section-card">')
        ui.html('<div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:14px;">Cross-Cluster Metric Comparison</div>')
        _bar_comparison_chart()
        ui.html('</div>')


def _build_insights(container):
    container.clear()
    with container:
        ui.html('<h1 class="page-title">Customer Insights</h1>')
        ui.html('<p class="page-sub">Data-derived commercial personas with actionable strategic playbooks.</p>')
        ui.separator().style("margin: 18px 0 24px 0; opacity: .25;")

        for c in sorted(_personas.keys()):
            p = _personas[c]
            row = p["row"]
            pct = (row["Count"] / _overall["total"]) * 100
            ui.html(f'''
            <div class="cluster-card" style="margin-bottom:18px;">
                <div style="position:absolute;left:0;top:0;bottom:0;width:5px;background:{p["color"]};border-radius:14px 0 0 14px;"></div>
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:14px;">
                    <div>
                        <span style="font-size:11px;font-weight:700;color:{p["color"]};text-transform:uppercase;letter-spacing:.05em;">CLUSTER {c}</span>
                        <div style="font-family:Plus Jakarta Sans,sans-serif;font-size:20px;font-weight:700;color:#172B4D;margin-top:2px;">{p["name"]}</div>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:20px;font-weight:700;color:#172B4D;">{int(row["Count"])}</span>
                        <span style="font-size:13px;color:#697586;"> customers ({pct:.1f}%)</span>
                    </div>
                </div>
                <div class="stat-grid" style="margin-bottom:14px;">
                    <div><div class="stat-label">Avg Age</div><div class="stat-val">{row["Age"]:.1f} yrs</div></div>
                    <div><div class="stat-label">Annual Income</div><div class="stat-val">${row["Annual Income (k$)"]:.1f}k</div></div>
                    <div><div class="stat-label">Spending Score</div><div class="stat-val">{row["Spending Score (1-100)"]:.1f}/100</div></div>
                </div>
                <div style="font-size:13.5px;color:#334155;line-height:1.55;">
                    <span style="font-weight:700;color:{p["color"]};">&#10148; Strategy:</span> {p["desc"]}
                </div>
            </div>
            ''')


def _build_prediction(container):
    container.clear()
    with container:
        ui.html('<h1 class="page-title">Customer Prediction</h1>')
        ui.html('<p class="page-sub">Score a new customer through the production K-Means pipeline.</p>')
        ui.separator().style("margin: 18px 0 24px 0; opacity: .25;")

        result_area = ui.column().style("width:100%;")

        with ui.row().style("width:100%; gap:24px; flex-wrap:wrap; align-items:flex-start;"):
            # Input form card
            with ui.column().style("flex:1; min-width:320px;"):
                with ui.card().style("border-radius:14px; padding:28px; border:1px solid #E5EAF0; box-shadow:0 1px 3px rgba(16,24,40,.04); width:100%;"):
                    ui.html('<div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:18px;">Customer Attributes</div>')
                    age_slider = ui.slider(min=18, max=75, value=32, step=1).style("width:100%;")
                    ui.html(f'<div style="font-size:12px;color:#697586;margin:-4px 0 12px 0;">Age (years)</div>')

                    income_slider = ui.slider(min=15, max=140, value=75, step=1).style("width:100%;")
                    ui.html(f'<div style="font-size:12px;color:#697586;margin:-4px 0 12px 0;">Annual Income (k$)</div>')

                    score_slider = ui.slider(min=1, max=100, value=50, step=1).style("width:100%;")
                    ui.html(f'<div style="font-size:12px;color:#697586;margin:-4px 0 18px 0;">Spending Score (1-100)</div>')

                    def do_predict():
                        a, i, s = age_slider.value, income_slider.value, score_slider.value
                        if a is None or i is None or s is None:
                            ui.notify("Please set all inputs.", type="warning")
                            return
                        X_new = pd.DataFrame([[a, i, s]], columns=FEATURE_COLUMNS)
                        X_sc  = pd.DataFrame(_scaler.transform(X_new), columns=FEATURE_COLUMNS)
                        pred  = int(_kmeans.predict(X_sc)[0])
                        p = _personas[pred]
                        row = p["row"]
                        result_area.clear()
                        with result_area:
                            ui.html(f'''
                            <div class="pred-result" style="margin-top:24px; border-top-color:{p["color"]};">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                                    <span class="badge">Predicted: Cluster {pred}</span>
                                    <span style="font-size:12px;color:#697586;">Pipeline inference</span>
                                </div>
                                <div style="font-family:Plus Jakarta Sans,sans-serif;font-size:22px;font-weight:700;color:#172B4D;margin-bottom:8px;">{p["name"]}</div>
                                <div class="stat-grid" style="margin-bottom:14px;">
                                    <div><div class="stat-label">Cluster Avg Age</div><div class="stat-val">{row["Age"]:.1f} yrs</div></div>
                                    <div><div class="stat-label">Cluster Avg Income</div><div class="stat-val">${row["Annual Income (k$)"]:.1f}k</div></div>
                                    <div><div class="stat-label">Cluster Avg Spending</div><div class="stat-val">{row["Spending Score (1-100)"]:.1f}/100</div></div>
                                </div>
                                <div style="background:#F8FAFC;border:1px solid #E5EAF0;border-radius:8px;padding:12px 16px;">
                                    <div style="font-size:12px;font-weight:700;color:{p["color"]};text-transform:uppercase;margin-bottom:4px;">Recommended Strategy</div>
                                    <div style="font-size:13px;color:#172B4D;line-height:1.5;">{p["desc"]}</div>
                                </div>
                                <div style="margin-top:14px;font-size:12px;color:#697586;">
                                    Input: Age={a}, Income=${i}k, Score={s}/100 &rarr; Cluster {pred} ({int(row["Count"])} members)
                                </div>
                            </div>
                            ''')
                        ui.notify(f"Classified as Cluster {pred}: {p['name']}", type="positive")

                    ui.button("Classify Customer", on_click=do_predict, icon="psychology").style(
                        "background:#0D9488 !important; color:#FFF; font-weight:600; border-radius:8px; width:100%; padding:10px;"
                    )


def _build_data_model(container):
    container.clear()
    with container:
        ui.html('<h1 class="page-title">Data & Model</h1>')
        ui.html('<p class="page-sub">Transparent evaluation diagnostics and verified cohort explorer.</p>')
        ui.separator().style("margin: 18px 0 24px 0; opacity: .25;")

        # Evaluation charts
        if _eval_df is not None:
            ui.html('<div class="section-card" style="margin-bottom:20px;">')
            ui.html('<div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:6px;">K-Means Model Selection Diagnostics</div>')
            ui.html('<div style="font-size:12px;color:#697586;margin-bottom:14px;">Evaluation across cluster candidates k = 2 through k = 10</div>')
            _evaluation_charts()
            ui.html('</div>')

            ui.html(f'''
            <div class="section-card" style="margin-bottom:20px;">
                <div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:8px;">Cluster Selection Justification (k = 6)</div>
                <p style="font-size:13px;color:#475569;line-height:1.55;margin:0;">
                    Across candidates k ∈ [2, 10], <strong>k = 6</strong> achieves the maximum
                    <strong>Silhouette Score of 0.4311</strong> and a minimised
                    <strong>Davies-Bouldin Index of 0.8350</strong>.
                    The Elbow curve shows clear inflection at k = 6. From a commercial perspective,
                    6 segments accurately resolve distinct demographic quadrants without over-segmentation.
                </p>
            </div>
            ''')

        # Pipeline info
        ui.html(f'''
        <div class="section-card" style="margin-bottom:20px;">
            <div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:8px;">Production Pipeline</div>
            <div style="font-family:monospace;font-size:13px;background:#F8FAFC;border:1px solid #E5EAF0;padding:14px;border-radius:8px;line-height:1.7;">
                <strong>Step 1:</strong> StandardScaler on [Age, Annual Income (k$), Spending Score (1-100)]<br>
                <strong>Step 2:</strong> KMeans (n_clusters=6, init=k-means++, random_state=42, n_init=auto)<br>
                <strong>Persistence:</strong> models/pipeline.joblib
            </div>
        </div>
        ''')

        # Dataset explorer
        ui.html('<div class="section-card">')
        ui.html('<div style="font-size:15px;font-weight:600;color:#172B4D;margin-bottom:14px;">Verified Customer Dataset (N = 200)</div>')
        columns = [
            {"name": c, "label": c, "field": c, "sortable": True, "align": "left"}
            for c in _df_raw.columns
        ]
        rows = _df_raw.to_dict("records")
        ui.table(columns=columns, rows=rows, row_key="CustomerID", pagination={"rowsPerPage": 15}).style("width:100%;")
        ui.html('</div>')


# ===================================================================
# CHART HELPERS  (ECharts via ui.echart)
# ===================================================================

def _kpi(label, value, sub, icon):
    ui.html(f'''
    <div class="kpi-card" style="flex:1;min-width:200px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div class="kpi-label">{label}</div>
            <div style="width:34px;height:34px;border-radius:8px;background:#F0FDFA;display:flex;align-items:center;justify-content:center;">
                <span class="material-icons" style="font-size:18px;color:#0D9488;">{icon}</span>
            </div>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    ''')


def _scatter_chart():
    series = []
    for c in sorted(_df_raw["Cluster"].unique()):
        cdata = _df_raw[_df_raw["Cluster"] == c]
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        name = _personas.get(c, {}).get("name", f"Cluster {c}")
        data = [[r["Annual Income (k$)"], r["Spending Score (1-100)"], r["Age"], r["CustomerID"]] for _, r in cdata.iterrows()]
        series.append({
            "name": f"C{c}: {name}",
            "type": "scatter",
            "data": data,
            "symbolSize": 10,
            "itemStyle": {"color": color, "borderColor": "#FFF", "borderWidth": 1},
        })
    ui.echart({
        "tooltip": {
            "trigger": "item",
            "formatter": """function(p){
                var d=p.data;
                return '<b>'+p.seriesName+'</b><br/>Income: $'+d[0]+'k<br/>Spending: '+d[1]+'/100<br/>Age: '+d[2]+' yrs<br/>ID: #'+d[3];
            }"""
        },
        "legend": {"bottom": 0, "textStyle": {"fontSize": 11, "color": "#697586"}},
        "xAxis": {"name": "Annual Income (k$)", "nameLocation": "center", "nameGap": 30, "splitLine": {"lineStyle": {"color": "#E5EAF0"}}},
        "yAxis": {"name": "Spending Score", "nameLocation": "center", "nameGap": 35, "splitLine": {"lineStyle": {"color": "#E5EAF0"}}},
        "series": series,
    }).style("height:420px; width:100%;")


def _age_scatter_chart():
    series = []
    for c in sorted(_df_raw["Cluster"].unique()):
        cdata = _df_raw[_df_raw["Cluster"] == c]
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        name = _personas.get(c, {}).get("name", f"Cluster {c}")
        data = [[r["Age"], r["Spending Score (1-100)"], r["Annual Income (k$)"], r["CustomerID"]] for _, r in cdata.iterrows()]
        series.append({
            "name": f"C{c}: {name}",
            "type": "scatter",
            "data": data,
            "symbolSize": 9,
            "itemStyle": {"color": color, "borderColor": "#FFF", "borderWidth": 1},
        })
    ui.echart({
        "tooltip": {
            "trigger": "item",
            "formatter": """function(p){
                var d=p.data;
                return '<b>'+p.seriesName+'</b><br/>Age: '+d[0]+' yrs<br/>Spending: '+d[1]+'/100<br/>Income: $'+d[2]+'k<br/>ID: #'+d[3];
            }"""
        },
        "legend": {"bottom": 0, "textStyle": {"fontSize": 11, "color": "#697586"}},
        "xAxis": {"name": "Age (Years)", "nameLocation": "center", "nameGap": 30, "splitLine": {"lineStyle": {"color": "#E5EAF0"}}},
        "yAxis": {"name": "Spending Score", "nameLocation": "center", "nameGap": 35, "splitLine": {"lineStyle": {"color": "#E5EAF0"}}},
        "series": series,
    }).style("height:400px; width:100%;")


def _donut_chart():
    data = []
    colors = []
    for c in sorted(_cluster_agg.index):
        name = _personas.get(c, {}).get("name", f"Cluster {c}")
        data.append({"value": int(_cluster_agg.loc[c, "Count"]), "name": f"C{c}: {name}"})
        colors.append(CLUSTER_COLORS[c % len(CLUSTER_COLORS)])
    ui.echart({
        "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c} customers ({d}%)"},
        "color": colors,
        "series": [{
            "type": "pie",
            "radius": ["50%", "78%"],
            "center": ["50%", "48%"],
            "avoidLabelOverlap": True,
            "label": {"show": True, "formatter": "{d}%", "fontSize": 11, "color": "#697586"},
            "itemStyle": {"borderRadius": 6, "borderColor": "#FFF", "borderWidth": 2},
            "data": data,
        }],
        "graphic": [{
            "type": "text",
            "left": "center", "top": "center",
            "style": {
                "text": str(_overall["total"]),
                "fontSize": 26, "fontWeight": "bold",
                "fill": "#172B4D", "fontFamily": "Plus Jakarta Sans, sans-serif",
            }
        }],
    }).style("height:340px; width:100%;")


def _bar_comparison_chart():
    clusters = sorted(_cluster_agg.index)
    cats = [f"C{c}" for c in clusters]
    ui.echart({
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0, "textStyle": {"fontSize": 11, "color": "#697586"}},
        "xAxis": {"type": "category", "data": cats, "axisLabel": {"fontSize": 11}},
        "yAxis": {"splitLine": {"lineStyle": {"color": "#E5EAF0"}}},
        "color": ["#2563EB", "#0D9488", "#F59E0B"],
        "series": [
            {"name": "Avg Age (yrs)", "type": "bar", "data": [round(_cluster_agg.loc[c, "Age"], 1) for c in clusters], "barGap": "10%"},
            {"name": "Avg Income ($k)", "type": "bar", "data": [round(_cluster_agg.loc[c, "Annual Income (k$)"], 1) for c in clusters]},
            {"name": "Avg Spending (1-100)", "type": "bar", "data": [round(_cluster_agg.loc[c, "Spending Score (1-100)"], 1) for c in clusters]},
        ],
    }).style("height:360px; width:100%;")


def _evaluation_charts():
    ks = _eval_df["k"].tolist()
    ui.echart({
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0, "textStyle": {"fontSize": 11, "color": "#697586"}},
        "grid": [
            {"left": "6%", "right": "54%", "bottom": "15%"},
            {"left": "56%", "right": "4%", "bottom": "15%"},
        ],
        "xAxis": [
            {"type": "category", "data": ks, "gridIndex": 0, "name": "k", "nameLocation": "center", "nameGap": 25},
            {"type": "category", "data": ks, "gridIndex": 1, "name": "k", "nameLocation": "center", "nameGap": 25},
        ],
        "yAxis": [
            {"type": "value", "gridIndex": 0, "name": "Inertia", "splitLine": {"lineStyle": {"color": "#E5EAF0"}}},
            {"type": "value", "gridIndex": 1, "name": "Score", "splitLine": {"lineStyle": {"color": "#E5EAF0"}}},
        ],
        "color": ["#172B4D", "#0D9488", "#E11D48"],
        "series": [
            {"name": "Inertia (SSE)", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0,
             "data": [round(v, 1) for v in _eval_df["inertia"]], "symbol": "circle", "symbolSize": 8,
             "lineStyle": {"width": 2.5},
             "markPoint": {"data": [{"xAxis": "6", "yAxis": round(_eval_df[_eval_df["k"]==6]["inertia"].values[0], 1),
                                     "symbol": "diamond", "symbolSize": 16, "itemStyle": {"color": "#0D9488"}}]}},
            {"name": "Silhouette (↑)", "type": "line", "xAxisIndex": 1, "yAxisIndex": 1,
             "data": [round(v, 4) for v in _eval_df["silhouette"]], "symbol": "circle", "symbolSize": 7,
             "lineStyle": {"width": 2.5}},
            {"name": "Davies-Bouldin (↓)", "type": "line", "xAxisIndex": 1, "yAxisIndex": 1,
             "data": [round(v, 4) for v in _eval_df["davies_bouldin"]], "symbol": "circle", "symbolSize": 6,
             "lineStyle": {"width": 2, "type": "dashed"}},
        ],
    }).style("height:380px; width:100%;")


# ===================================================================
# MAIN LAYOUT
# ===================================================================

NAV_ITEMS = [
    ("Overview",               "dashboard",     _build_overview),
    ("Segmentation Explorer",  "scatter_plot",  _build_segmentation),
    ("Customer Insights",      "people",        _build_insights),
    ("Customer Prediction",    "psychology",     _build_prediction),
    ("Data & Model",           "storage",       _build_data_model),
]

@ui.page("/")
def main_page():
    ui.add_head_html(f"<style>{_GLOBAL_CSS}</style>")
    ui.add_head_html('<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">')

    # State
    current_page = {"idx": 0}

    # Content area reference
    content = ui.column()

    def navigate(idx):
        current_page["idx"] = idx
        # Update active states
        for i, btn in enumerate(nav_buttons):
            if i == idx:
                btn.classes(add="active-nav")
            else:
                btn.classes(remove="active-nav")
        # Update topbar title
        topbar_title.set_text(NAV_ITEMS[idx][0])
        # Rebuild content
        NAV_ITEMS[idx][2](content)

    # -- Sidebar --
    with ui.left_drawer(value=True, fixed=True, bordered=False).style("width:260px;").classes("q-pa-none") as drawer:
        # Brand
        with ui.column().style("padding:18px 18px 14px 18px; border-bottom:1px solid #E2E8F0;"):
            with ui.row().style("align-items:center; gap:10px;"):
                ui.html('<div style="width:34px;height:34px;border-radius:8px;background:#172B4D;display:flex;align-items:center;justify-content:center;color:#FFF;font-weight:700;font-size:13px;letter-spacing:-0.02em;"><span style="color:#0F766E;font-weight:800;">C</span>I</div>')
                with ui.column().style("gap:0;"):
                    ui.label("CUSTOMER INTELLIGENCE").style("font-size:12px; font-weight:700; letter-spacing:.02em; color:#172B4D; line-height:1.2;")
                    ui.label("Segmentation & Analytics").style("font-size:10px; font-weight:500; color:#64748B; text-transform:uppercase; letter-spacing:.04em;")

        # Nav items
        nav_buttons = []
        for i, (label, icon, _builder) in enumerate(NAV_ITEMS):
            item = ui.item(label, on_click=lambda idx=i: navigate(idx)).style("cursor:pointer;")
            item.props('clickable')
            with item:
                with item.add_slot("prepend"):
                    ui.icon(icon).style("font-size:18px; margin-right:8px;")
            if i == 0:
                item.classes("active-nav")
            nav_buttons.append(item)

        # Spacer + status
        ui.space()
        with ui.column().style("padding:12px 14px; margin:12px; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; box-shadow:0 1px 2px rgba(15,23,42,0.03);"):
            with ui.row().style("align-items:center; gap:6px;"):
                ui.html('<span style="width:7px;height:7px;border-radius:50%;background:#0F766E;display:inline-block;"></span>')
                ui.label("K-Means Engine Active").style("font-size:11px; font-weight:600; color:#172B4D;")
            ui.label("k=6 · Silhouette 0.43 · 200 records").style("font-size:10px; color:#64748B; margin-top:2px;")

    # -- Top bar --
    with ui.header(elevated=False).style("background:#F5F7FA; border-bottom:1px solid #E2E8F0; padding:0 28px; height:56px;"):
        with ui.row().style("width:100%; align-items:center; justify-content:space-between; height:100%;"):
            with ui.row().style("align-items:center; gap:12px;"):
                ui.button(icon="menu", on_click=drawer.toggle).props("flat dense round").style("color:#172B4D;")
                topbar_title = ui.label("Overview").style("font-family:Plus Jakarta Sans,sans-serif; font-size:17px; font-weight:700; color:#172B4D;")
            ui.html('<span class="badge">ML Intelligence Platform</span>')

    # -- Main content area --
    content.style("padding:28px; width:100%;")

    # Build initial page
    navigate(0)


# ===================================================================
# RUN
# ===================================================================
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Customer Intelligence Studio",
        host="0.0.0.0",
        port=8080,
        reload=False,
        show=True,
    )

