import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time

# PAGE CONFIG

st.set_page_config(
    page_title="Retail Visitor Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = Path("output")
LIVE_JSON = OUTPUT_DIR / "live_counts.json"
HISTORY_CSV = OUTPUT_DIR / "history.csv"

AGE_ORDER = ["0-12", "13-24", "25-39", "40-59", "60+"]


# CUSTOM CSS
st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 4rem;
        padding-bottom: 1rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 1600px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f3f6fb 100%);
        border-right: 1px solid #e6ebf2;
    }

    .header-card {
        background: white;
        border: 1px solid #e7ecf3;
        border-radius: 18px;
        padding: 1.1rem 1.2rem 0.9rem 1.2rem;
        box-shadow: 0 4px 14px rgba(20, 28, 45, 0.06);
        margin-bottom: 1rem;
    }

    .dashboard-title {
        font-size: clamp(1.8rem, 2.8vw, 2.7rem);
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.35rem;
        line-height: 1.15;
    }

    .dashboard-subtitle {
        font-size: clamp(0.95rem, 1.15vw, 1rem);
        color: #5b6472;
        margin-bottom: 0;
    }

    .card {
        background: white;
        padding: 1rem 1rem 0.7rem 1rem;
        border-radius: 18px;
        border: 1px solid #e7ecf3;
        box-shadow: 0 4px 14px rgba(20, 28, 45, 0.06);
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: clamp(1rem, 1.2vw, 1.08rem);
        font-weight: 700;
        color: #243041;
        margin-bottom: 0.65rem;
    }

    .small-note {
        font-size: 0.85rem;
        color: #6b7280;
    }

    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e7ecf3;
        padding: 1rem 1rem;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(20, 28, 45, 0.06);
        min-height: 120px;
    }

    div[data-testid="stMetricLabel"] {
    font-weight: 900 !important;
    font-size: 2.5rem !important;
    color: #1f2937 !important;
}

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: clamp(1.7rem, 2.6vw, 2.5rem);
        color: #2563EB;
        font-weight: 800;
    }

    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.95rem;
        color: #10B981 !important;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 12px;
        width: 100%;
        font-weight: 600;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        div[data-testid="metric-container"] {
            min-height: 105px;
        }
    }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("## Dashboard Filters")

location = st.sidebar.selectbox(
    "Select Store Location",
    ["Mannheim", "Heidelberg", "Berlin", "Stuttgart", "Munich"]
)

date_range = st.sidebar.date_input("Date Range")

zones = st.sidebar.multiselect(
    "Store Zones",
    [
        "Entrance",
        "Aisle 1 - Fresh Produce & Floral",
        "Aisle 2 - Bakery",
        "Aisle 3 - Deli & Prepared Foods",
        "Aisle 4 - Electronics",
        "Aisle 5 - Frozen Foods",
        "Aisle 6 - Kid-Level Items",
        "Aisle 7 - Snacks & Chips",
        "Checkout",
    ],
    default=["Entrance"]
)

if st.sidebar.button("Apply Filters", use_container_width=True):
    st.sidebar.success("Filters applied")


# DATA LOADING
def load_data():
    if not LIVE_JSON.exists() or not HISTORY_CSV.exists():
        return None, None

    with open(LIVE_JSON, "r") as f:
        live_data = json.load(f)

    history_df = pd.read_csv(HISTORY_CSV)
    history_df["timestamp_iso"] = pd.to_datetime(history_df["timestamp_iso"], errors="coerce")
    history_df = history_df.dropna(subset=["timestamp_iso"]).sort_values("timestamp_iso").reset_index(drop=True)

    return live_data, history_df


def get_dynamic_delta(history_df, age_cols):
    if history_df is None or history_df.empty or not age_cols:
        return "No history"

    valid_df = history_df.dropna(subset=age_cols, how="all").copy()
    if len(valid_df) == 0:
        return "No history"

    current_total = valid_df.iloc[-1][age_cols].fillna(0).sum()

    if len(valid_df) < 2:
        return "First window"

    previous_total = valid_df.iloc[-2][age_cols].fillna(0).sum()

    if previous_total == 0:
        if current_total == 0:
            return "0.0% vs prev window"
        return "New traffic vs prev window"

    percent_change = ((current_total - previous_total) / previous_total) * 100
    return f"{percent_change:+.1f}% vs prev window"


def get_previous_total(history_df, age_cols):
    if history_df is None or history_df.empty or not age_cols:
        return 0

    valid_df = history_df.dropna(subset=age_cols, how="all").copy()
    if len(valid_df) == 0:
        return 0
    if len(valid_df) < 2:
        return int(valid_df.iloc[-1][age_cols].fillna(0).sum())

    return int(valid_df.iloc[-2][age_cols].fillna(0).sum())


live_data, history_df = load_data()

# HEADER

# st.markdown('<div class="header-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-title">Privacy-Preserving Retail Demographics Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div class="dashboard-subtitle">Only aggregated age-bucket counts are stored · Location: <b>{location}</b></div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)


if live_data is not None and history_df is not None:
    age_cols = [c for c in AGE_ORDER if c in history_df.columns]

    total_counts = live_data.get("total_counts", {})
    total_visitors = int(sum(total_counts.values())) if total_counts else 0
    peak_age = max(total_counts, key=total_counts.get) if total_counts else "N/A"
    dynamic_delta = get_dynamic_delta(history_df, age_cols)
    previous_total = get_previous_total(history_df, age_cols)

    
    # KPI ROW

    k1, k2, k3 = st.columns(3)

    with k1:
        live_counter_placeholder = st.empty()

        steps = 12
        start_value = previous_total if previous_total <= total_visitors else 0
        for i in range(1, steps + 1):
            animated_value = int(start_value + (total_visitors - start_value) * (i / steps))
            live_counter_placeholder.metric(
                "Total Visitors",
                f"{animated_value:,}",
                delta=dynamic_delta
            )
            time.sleep(0.03)

    with k2:
        st.metric("Peak Age Group", peak_age)

    with k3:
        st.metric("Selected Zones", len(zones) if zones else 0)


    
    # AGE DISTRIBUTION

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Age Distribution (Total Store)</div>', unsafe_allow_html=True)

    dist_df = pd.DataFrame({
        "Age Group": list(total_counts.keys()),
        "Count": list(total_counts.values())
    })

    if not dist_df.empty:
        dist_df["Age Group"] = pd.Categorical(
            dist_df["Age Group"],
            categories=AGE_ORDER,
            ordered=True
        )
        dist_df = dist_df.sort_values("Age Group")

        fig_bar = px.bar(
            dist_df,
            x="Age Group",
            y="Count",
            text="Count",
            template="plotly_white"
        )

        fig_bar.update_traces(
            marker_color="#2563EB",
            textposition="outside",
            hovertemplate="Age Group: %{x}<br>Count: %{y}<extra></extra>"
        )

        fig_bar.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Age Group",
            yaxis_title="Visitors",
            plot_bgcolor="white",
            paper_bgcolor="white",
            bargap=0.35
        )

        fig_bar.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
        st.plotly_chart(fig_bar, use_container_width=True, config={"responsive": True})
    else:
        st.info("No live distribution data available.")

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # TREND + HOTSPOTS
    # --------------------------------------------------
    col_a, col_b = st.columns([2.1, 1])

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Age Demographic Trend Over Time</div>', unsafe_allow_html=True)

        if age_cols:
            trend_df = history_df.copy()

            fig_line = go.Figure()

            color_map = {
                "0-12": "#2563EB",
                "13-24": "#60A5FA",
                "25-39": "#10B981",
                "40-59": "#F59E0B",
                "60+": "#EF4444",
            }

            for age in age_cols:
                fig_line.add_trace(
                    go.Scatter(
                        x=trend_df["timestamp_iso"],
                        y=trend_df[age],
                        mode="lines",
                        name=age,
                        line=dict(width=2.2, color=color_map.get(age)),
                        hovertemplate=(
                            f"Age Group: {age}<br>"
                            "Time: %{x}<br>"
                            "Count: %{y}<extra></extra>"
                        )
                    )
                )

            fig_line.update_layout(
                template="plotly_white",
                height=420,
                margin=dict(l=10, r=10, t=10, b=75),
                xaxis_title="Timestamp",
                yaxis_title="Visitors",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="left",
                    x=0
                ),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            fig_line.update_xaxes(
                tickangle=-45,
                showgrid=False,
                tickfont=dict(size=10)
            )

            fig_line.update_yaxes(
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                zeroline=False
            )

            st.plotly_chart(fig_line, use_container_width=True, config={"responsive": True})
        else:
            st.warning("Required age columns not found in history.csv")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Visitor Density by Age Group & Zone</div>', unsafe_allow_html=True)

        if total_counts:
            hotspot_df = pd.DataFrame({
                "Age Group": list(total_counts.keys()),
                "Traffic": list(total_counts.values())
            })

            hotspot_df["Age Group"] = pd.Categorical(
                hotspot_df["Age Group"],
                categories=AGE_ORDER,
                ordered=True
            )
            hotspot_df = hotspot_df.sort_values("Age Group")

            selected_zones = zones if zones else ["Main"]
            heatmap_rows = []

            for _zone in selected_zones:
                heatmap_rows.append(hotspot_df["Traffic"].tolist())

            fig_heat = px.imshow(
                heatmap_rows,
                labels=dict(x="Age Group", y="Zone", color="Traffic"),
                x=hotspot_df["Age Group"].astype(str).tolist(),
                y=selected_zones,
                color_continuous_scale=["#EFF6FF", "#BFDBFE", "#60A5FA", "#2563EB"],
                aspect="auto"
            )

            fig_heat.update_layout(
                template="plotly_white",
                height=max(260, 120 + len(selected_zones) * 55),
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_colorbar=dict(title="Count")
            )

            st.plotly_chart(fig_heat, use_container_width=True, config={"responsive": True})

            st.markdown(
                f'<div class="small-note">Selected zones: <b>{", ".join(selected_zones)}</b></div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No hotspot data available.")

        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # INSIGHTS + TABLE
    # --------------------------------------------------
    c1, c2 = st.columns([1.15, 1.85])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Quick Insights</div>', unsafe_allow_html=True)

        if total_counts:
            sorted_counts = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
            top_1 = sorted_counts[0][0] if len(sorted_counts) > 0 else "N/A"
            top_2 = sorted_counts[1][0] if len(sorted_counts) > 1 else "N/A"

            st.markdown(f"""
**Top segment:** {top_1}  
**Second strongest segment:** {top_2}  
**Current store traffic:** {total_visitors} visitors  
**Primary selected zone:** {zones[0] if zones else "Main"}  

This dashboard stores only **aggregated age-bucket counts**, which supports a privacy-preserving retail analytics workflow.
""")
        else:
            st.write("No insights available.")

        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Raw Demographic Data</div>', unsafe_allow_html=True)

        if not dist_df.empty:
            raw_table = dist_df.copy()
            raw_table["Age Group"] = raw_table["Age Group"].astype(str)
            st.dataframe(raw_table, use_container_width=True, hide_index=True)
        else:
            st.write("No data available.")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning(
        "Waiting for data from edge.py... Make sure the script is running and generating "
        "'output/live_counts.json' and 'output/history.csv'."
    )
    if st.button("Retry"):
        st.rerun()

# AUTO REFRESH

time.sleep(5)
st.rerun()



