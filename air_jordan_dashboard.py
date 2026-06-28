"""
Air Jordan Resale Intelligence Dashboard
=========================================
Dark theme matching dashboard-9 aesthetics:
  - Pure black/charcoal backgrounds (#0d0d0d, #111, #1a1a1a)
  - White/light grey text
  - Segmented bar charts (stacked, hatched-look via opacity layers)
  - Funnel chart (conversion-style)
  - Heatmap grid (sales-by-hour style)
  - Traffic source bars (dotted progress bars)
  - Monochrome palette with a single green accent for positive deltas
Run:  streamlit run air_jordan_dashboard_dark.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io

# ─────────────────────────────────────────────
# DARK THEME PALETTE  (dashboard-9 inspired)
# ─────────────────────────────────────────────
BG_PAGE    = "#0d0d0d"
BG_CARD    = "#111111"
BG_CARD2   = "#161616"
BORDER     = "#2a2a2a"
TEXT_PRI   = "#f1f1f1"
TEXT_SEC   = "#888888"
TEXT_MUTED = "#555555"
ACCENT_GRN = "#4ade80"   # positive delta green
ACCENT_RED = "#f87171"   # negative delta red
GREYS      = ["#2e2e2e", "#444444", "#666666", "#888888", "#aaaaaa", "#cccccc", "#eeeeee"]
PLOTLY_FONT = dict(family="'DM Mono', 'Courier New', monospace", color=TEXT_PRI)

GLOBAL_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=PLOTLY_FONT,
    margin=dict(l=12, r=12, t=28, b=12),
)

# Default axis style — merge into update_layout calls that don't override axes
_AXIS_DEFAULTS = dict(
    showgrid=False, zeroline=False, showline=False,
    tickfont=dict(size=10, color=TEXT_SEC), color=TEXT_SEC,
)

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Air Jordan Resale Intelligence",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# GLOBAL CSS  — dark cards, mono font, tight borders
# ══════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color: {BG_PAGE} !important;
    color: {TEXT_PRI};
    font-family: 'DM Sans', sans-serif;
}}
[data-testid="stSidebar"][aria-expanded="true"] {{
    background-color: #0a0a0a !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"][aria-expanded="true"] * {{ color: {TEXT_PRI} !important; }}
[data-testid="stSidebar"][aria-expanded="true"] .stSelectbox label,
[data-testid="stSidebar"][aria-expanded="true"] .stDateInput label {{
    color: {TEXT_SEC} !important; font-size: 0.75rem !important;
}}
section[data-testid="stSidebar"][aria-expanded="true"] {{ padding-top: 1rem; }}
.stFileUploader {{ background: {BG_CARD} !important; border: 1px solid {BORDER} !important; border-radius: 6px; }}
.stFileUploader label {{ color: {TEXT_SEC} !important; font-size: 0.8rem !important; }}

/* cards */
.card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 18px 20px;
    position: relative;
}}
.card-corner {{
    position: absolute;
    width: 8px; height: 8px;
    border-color: {TEXT_MUTED};
    border-style: solid;
}}
.card-tl {{ top: -1px; left: -1px; border-width: 1px 0 0 1px; }}
.card-tr {{ top: -1px; right: -1px; border-width: 1px 1px 0 0; }}
.card-bl {{ bottom: -1px; left: -1px; border-width: 0 0 1px 1px; }}
.card-br {{ bottom: -1px; right: -1px; border-width: 0 1px 1px 0; }}

/* st.container(border=True) — styled to match dark card theme */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    padding: 16px 18px !important;
}}

/* KPI metric cards */
.kpi-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 16px 18px 14px;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}}
/* Force KPI row columns to stretch to equal height */
[data-testid="stHorizontalBlock"] [data-testid="column"] > div:first-child {{
    height: 100%;
    display: flex;
    flex-direction: column;
}}
.kpi-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: {TEXT_SEC};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}}
.kpi-value {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: {TEXT_PRI};
    line-height: 1;
}}
.kpi-delta-pos {{
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: {ACCENT_GRN};
    margin-top: 6px;
}}
.kpi-delta-neg {{
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: {ACCENT_RED};
    margin-top: 6px;
}}
.kpi-sub {{
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    margin-top: 2px;
}}

/* section headers */
.sec-header {{
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: {TEXT_SEC};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin-bottom: 14px;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
    gap: 5px;
}}
.sec-badge {{
    font-size: 0.62rem;
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    padding: 1px 6px;
    border-radius: 2px;
    letter-spacing: 0.06em;
}}

/* progress bar traffic source rows */
.ts-row {{ margin-bottom: 14px; }}
.ts-label {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
.ts-name {{ font-size: 0.82rem; color: {TEXT_PRI}; }}
.ts-count {{ font-size: 0.82rem; color: {TEXT_SEC}; font-family: 'DM Mono', monospace; }}
.ts-bar-bg {{
    background: {BG_CARD2}; border-radius: 1px; height: 4px; width: 100%;
}}
.ts-bar-fill {{
    background: {TEXT_SEC}; border-radius: 1px; height: 4px;
    background-image: repeating-linear-gradient(
        90deg, {TEXT_SEC} 0px, {TEXT_SEC} 3px, transparent 3px, transparent 5px
    );
}}

/* quick actions */
.qa-item {{
    display: flex; align-items: flex-start; gap: 12px;
    padding: 10px 0; border-bottom: 1px solid {BORDER};
    cursor: pointer;
}}
.qa-item:last-child {{ border-bottom: none; }}
.qa-icon {{ font-size: 0.9rem; color: {TEXT_SEC}; margin-top: 1px; width: 20px; }}
.qa-text {{ flex: 1; }}
.qa-title {{ font-size: 0.85rem; color: {TEXT_PRI}; font-weight: 600; }}
.qa-sub {{ font-size: 0.75rem; color: {TEXT_SEC}; margin-top: 1px; }}
.qa-arrow {{ color: {TEXT_MUTED}; font-size: 0.8rem; margin-top: 2px; }}

/* stat tables */
.stat-table {{ width: 100%; border-collapse: collapse; }}
.stat-table th {{
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;
    text-align: left; padding: 0 0 8px; border-bottom: 1px solid {BORDER};
    font-weight: 400;
}}
.stat-table td {{ padding: 8px 0; border-bottom: 1px solid #1e1e1e; font-size: 0.83rem; }}
.stat-table tr:last-child td {{ border-bottom: none; }}

/* dataframe overrides */
[data-testid="stDataFrame"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px;
}}
.stDataFrame th {{
    background: #1a1a1a !important;
    color: {TEXT_SEC} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
}}
.stDataFrame td {{
    color: {TEXT_PRI} !important;
    font-size: 0.8rem !important;
    background: {BG_CARD} !important;
}}

/* hide streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}
.stApp > header {{ height: 0; }}
/* re-expand button lives in the header — make it visible and clickable */
[data-testid="stExpandSidebarButton"] {{
    visibility: visible !important;
    position: fixed !important;
    top: 0.5rem !important;
    left: 0.5rem !important;
    z-index: 999999 !important;
}}
div.block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1400px; }}

/* plotly chart container */
[data-testid="stPlotlyChart"] > div {{ border-radius: 0 !important; }}

/* metric overrides */
[data-testid="stMetricValue"] {{
    font-size: 1.6rem !important; font-weight: 700 !important; color: {TEXT_PRI} !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.7rem !important; color: {TEXT_SEC} !important;
    font-family: 'DM Mono', monospace !important; text-transform: uppercase;
}}
[data-testid="stMetricDelta"] svg {{ display: none; }}
[data-testid="stMetricDeltaIcon-Up"] {{ color: {ACCENT_GRN} !important; }}
[data-testid="stMetricDeltaIcon-Down"] {{ color: {ACCENT_RED} !important; }}

/* expander */
.streamlit-expanderHeader {{
    background: {BG_CARD} !important; color: {TEXT_SEC} !important;
    font-size: 0.8rem !important; border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
}}
.streamlit-expanderContent {{
    background: {BG_CARD} !important; border: 1px solid {BORDER} !important;
    border-top: none !important;
}}

/* tabs */
.stTabs [role="tablist"] {{ border-bottom: 1px solid {BORDER}; gap: 0; }}
.stTabs [role="tab"] {{
    background: transparent !important; color: {TEXT_MUTED} !important;
    font-size: 0.78rem !important; font-family: 'DM Mono', monospace !important;
    border: none !important; padding: 6px 16px !important; border-radius: 0 !important;
}}
.stTabs [aria-selected="true"] {{
    color: {TEXT_PRI} !important; border-bottom: 1px solid {TEXT_PRI} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ background: transparent !important; padding-top: 12px; }}

/* buttons */
.stButton > button {{
    background: {BG_CARD2} !important; color: {TEXT_PRI} !important;
    border: 1px solid {BORDER} !important; border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 0.05em;
}}
.stButton > button:hover {{
    background: #2a2a2a !important; border-color: #444 !important;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HELPER COMPONENTS
# ══════════════════════════════════════════════
def card(content_fn, *args, **kwargs):
    st.markdown('<div class="card"><div class="card-corner card-tl"></div>'
                '<div class="card-corner card-tr"></div>'
                '<div class="card-corner card-bl"></div>'
                '<div class="card-corner card-br"></div>', unsafe_allow_html=True)
    content_fn(*args, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)

def kpi_card(label, value, delta=None, sub=None, delta_pos=True):
    delta_class = "kpi-delta-pos" if delta_pos else "kpi-delta-neg"
    delta_arrow = "↑" if delta_pos else "↓"
    delta_html = f'<div class="{delta_class}">{delta_arrow} {delta}</div>' if delta else ""
    sub_html   = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}{sub_html}
    </div>""", unsafe_allow_html=True)

def sec_header(title, badge=None):
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ""
    st.markdown(f'<div class="sec-header"><span>{title}</span>{badge_html}</div>',
                unsafe_allow_html=True)

def traffic_bar(name, count, max_count):
    pct = int(count / max_count * 100) if max_count else 0
    count_str = f"{count/1000:.1f}K" if count >= 1000 else str(count)
    st.markdown(f"""
    <div class="ts-row">
      <div class="ts-label">
        <span class="ts-name">{name}</span>
        <span class="ts-count">{count_str}</span>
      </div>
      <div class="ts-bar-bg">
        <div class="ts-bar-fill" style="width:{pct}%"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# DATA PIPELINE (unchanged logic from original)
# ══════════════════════════════════════════════
@st.cache_data(show_spinner="Cleaning data …")
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes)) if file_bytes else pd.DataFrame()
    df.columns = (df.columns.str.strip().str.lower()
                    .str.replace(r"[\s/\-]+", "_", regex=True))
    aliases = {
        "sneaker_name":      ["name","model","shoe","sneaker","title","product","shoe_model"],
        "brand":             ["brand","manufacturer","make"],
        "colorway":          ["colorway","colour","color"],
        "release_date":      ["release_date","release","date_released","launch_date"],
        "retail_price":      ["retail_price","retail","msrp","original_price","retail_price_usd"],
        "resale_price":      ["resale_price","resale","sale_price","market_price","resale_price_usd"],
        "size":              ["size","shoe_size","us_size"],
        "sale_date":         ["sale_date","sold_date","transaction_date","date_sold"],
        "platform":          ["platform","marketplace","source","site","sales_channel"],
        "sales_volume":      ["sales_volume","number_of_sales","quantity","num_sales"],
        "profit_margin_pct": ["profit_margin_usd","profit_margin"],
    }
    for canonical, variants in aliases.items():
        for col in df.columns:
            if col in variants and canonical not in df.columns:
                df.rename(columns={col: canonical}, inplace=True)
    df.drop_duplicates(inplace=True)
    for dcol in ["release_date", "sale_date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce", format="mixed")
    if "sale_date" in df.columns:
        df["flag_bad_date"] = df["sale_date"].isna().astype(int)
    for nc in ["retail_price", "resale_price", "sales_volume", "size"]:
        if nc in df.columns:
            df[nc] = pd.to_numeric(
                df[nc].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    if "retail_price" in df.columns:
        df["retail_price"].fillna(df["retail_price"].median(), inplace=True)
    if "resale_price" in df.columns:
        df.dropna(subset=["resale_price"], inplace=True)
    if "sales_volume" in df.columns:
        df["sales_volume"].fillna(0, inplace=True)
    for cat in ["brand", "sneaker_name", "colorway", "platform"]:
        if cat in df.columns:
            df[cat].fillna("Unknown", inplace=True)
            df[cat] = df[cat].str.strip().str.title()
    if "resale_price" in df.columns:
        Q1, Q3 = df["resale_price"].quantile(0.25), df["resale_price"].quantile(0.75)
        IQR = Q3 - Q1; upper = Q3 + 3.0 * IQR
        df["resale_price_raw"] = df["resale_price"].copy()
        df["resale_price"] = df["resale_price"].clip(upper=upper)
        df["flag_outlier"] = (df["resale_price_raw"] > upper).astype(int)
    if {"retail_price", "resale_price"}.issubset(df.columns):
        if "profit_margin_pct" not in df.columns:
            df["profit_margin_pct"] = ((df["resale_price"] - df["retail_price"])
                                       / df["retail_price"] * 100).round(2)
        df["premium_usd"] = (df["resale_price"] - df["retail_price"]).round(2)
    if "release_date" in df.columns:
        today = pd.Timestamp.today()
        df["age_days"] = (today - df["release_date"]).dt.days
        df["age_bucket"] = pd.cut(df["age_days"], bins=[-1, 90, 365, 730, 99999],
            labels=["New Drop (<3 mo)","Recent (3-12 mo)","Established (1-2 yr)","Classic (2+ yr)"])
    elif "days_in_inventory" in df.columns:
        df["age_bucket"] = pd.cut(df["days_in_inventory"], bins=[-1, 30, 90, 180, 99999],
            labels=["Fast Flip (<30d)","Short Hold (30-90d)","Medium Hold (90-180d)","Long Hold (180d+)"])
    if "sale_date" in df.columns:
        df["sale_month"]     = df["sale_date"].dt.to_period("M")
        df["sale_year"]      = df["sale_date"].dt.year
        df["sale_month_num"] = df["sale_date"].dt.month
    return df


def add_segments(df):
    if "sales_volume" in df.columns:
        df_s = df.sort_values("sales_volume", ascending=False).copy()
        df_s["cum_pct"] = df_s["sales_volume"].cumsum() / df_s["sales_volume"].sum() * 100
        df_s["abc_class"] = df_s["cum_pct"].apply(
            lambda p: "A — Top sellers" if p <= 70 else ("B — Mid sellers" if p <= 90 else "C — Slow movers"))
        df = df.merge(df_s[["abc_class"]], left_index=True, right_index=True, how="left")
    cluster_cols = [c for c in ["resale_price", "profit_margin_pct"] if c in df.columns]
    if len(cluster_cols) == 2:
        sub = df[cluster_cols].dropna()
        X   = StandardScaler().fit_transform(sub)
        km  = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        df.loc[sub.index, "price_tier"] = labels
        tier_means = df.groupby("price_tier")["resale_price"].mean().sort_values()
        df["price_tier"] = df["price_tier"].map(
            {tier_means.index[0]: "Budget Tier",
             tier_means.index[1]: "Mid Tier",
             tier_means.index[2]: "Premium Tier"})
    return df


def descriptive_stats(df):
    num = df.select_dtypes(include="number")
    s   = num.agg(["mean","median","std","skew"]).T.round(3)
    s.columns = ["Mean","Median","Std Dev","Skewness"]
    return s


def _col_gradient(s):
    """Per-column CSS gradient that mirrors 'binary' cmap — no matplotlib needed."""
    mn, mx = s.min(), s.max()
    rng = mx - mn if mx != mn else 1.0
    def _css(v):
        norm = (v - mn) / rng          # 0.0 = darkest, 1.0 = lightest
        i = int(17 + norm * 30)        # #111111 → #2f2f2f
        return f"background-color: #{i:02x}{i:02x}{i:02x}; color: #f1f1f1;"
    return [_css(v) for v in s]


def mom_yoy(df):
    if "sale_month" not in df.columns:
        return pd.DataFrame()
    m = "resale_price" if "resale_price" in df.columns else df.select_dtypes("number").columns[0]
    monthly = df.groupby("sale_month")[m].mean().reset_index().sort_values("sale_month")
    monthly["MoM_%"] = monthly[m].pct_change() * 100
    monthly["YoY_%"] = monthly[m].pct_change(12) * 100
    return monthly.round(2)


def run_anova(df):
    mc = next((c for c in ["profit_margin_pct","profit_margin_usd","profit_margin"] if c in df.columns), None)
    bc = next((c for c in ["age_bucket","days_in_inventory"] if c in df.columns), None)
    if not mc or not bc:
        return None, None, None, None
    groups = [g[mc].dropna().values for _, g in df.groupby(bc, observed=True) if len(g) > 1]
    if len(groups) < 2:
        return None, None, None, None
    f, p = stats.f_oneway(*groups)
    return round(f, 4), round(p, 4), mc, bc


# ══════════════════════════════════════════════
# DARK PLOTLY HELPERS
# ══════════════════════════════════════════════
def dark_fig(fig, height=260):
    fig.update_layout(**GLOBAL_LAYOUT, height=height,
        xaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(size=9, color=TEXT_MUTED)),
        yaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(size=9, color=TEXT_MUTED)),
    )
    fig.update_traces(hovertemplate="%{y:,.0f}<extra></extra>")
    return fig


def segmented_bar_chart(categories, values, title="", height=260):
    """Stacked-segment style bar — each bar split into light/dark halves like dashboard-9"""
    n = len(categories)
    # create two layers: lighter top portion, darker bottom
    vals_bot = [v * 0.55 for v in values]
    vals_top = [v * 0.45 for v in values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=vals_bot, name="",
        marker=dict(color=GREYS[1], line=dict(width=0)),
        width=0.55,
    ))
    fig.add_trace(go.Bar(
        x=categories, y=vals_top, name="",
        marker=dict(color=GREYS[3], line=dict(width=0)),
        width=0.55,
        base=vals_bot,
        hovertemplate="%{base:,.0f}<extra></extra>",
    ))
    fig.update_layout(**GLOBAL_LAYOUT, height=height,
        barmode="stack", showlegend=False,
        title=dict(text=title, font=dict(size=10, color=TEXT_SEC), x=0),
        yaxis=dict(showgrid=True, gridcolor="#1e1e1e", zeroline=False,
                   tickfont=dict(size=9, color=TEXT_MUTED),
                   tickformat="$,.0f"),
        xaxis=dict(tickfont=dict(size=9, color=TEXT_MUTED), showgrid=False),
    )
    return fig


def funnel_chart(stages, values, height=320):
    """Horizontal funnel / conversion chart matching dashboard-9 style"""
    n = len(stages)
    max_v = max(values)
    fig = go.Figure()
    grey_scale = ["#555", "#444", "#333", "#222"]
    for i, (stage, val) in enumerate(zip(stages, values)):
        pct = val / max_v
        color = grey_scale[i] if i < len(grey_scale) else "#1a1a1a"
        fig.add_trace(go.Bar(
            x=[val], y=[i], orientation="h", name=stage,
            marker=dict(color=color, line=dict(width=0)),
            width=0.6,
            customdata=[[val/1000, val/max_v*100]],
            hovertemplate=f"<b>{stage}</b><br>%{{customdata[0]:.1f}}K  (%{{customdata[1]:.0f}}%)<extra></extra>",
        ))
    fig.update_layout(**GLOBAL_LAYOUT, height=height,
        barmode="overlay", showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(ticktext=stages, tickvals=list(range(n)),
                   tickfont=dict(size=10, color=TEXT_SEC), autorange="reversed",
                   showgrid=False, zeroline=False),
    )
    return fig


def donut_chart(labels, values, height=280):
    """Donut / pie with hatched grey fill aesthetic"""
    colors = [GREYS[2], GREYS[4], GREYS[0], GREYS[5], GREYS[1]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.0,
        marker=dict(colors=colors[:len(labels)],
                    line=dict(color=BG_PAGE, width=2)),
        textinfo="none",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(**GLOBAL_LAYOUT, height=height, showlegend=False)
    return fig


def heatmap_grid(df_matrix, height=260):
    """Calendar-style heatmap like 'sales by hour' grid"""
    fig = go.Figure(go.Heatmap(
        z=df_matrix.values,
        x=[str(c) for c in df_matrix.columns],
        y=[str(r) for r in df_matrix.index],
        colorscale=[[0, "#1a1a1a"], [0.3, "#333"], [0.7, "#666"], [1.0, "#bbb"]],
        showscale=False,
        xgap=2, ygap=2,
        hovertemplate="Hour: %{x}<br>Day: %{y}<br>Value: %{z:,.0f}<extra></extra>",
    ))
    # Build layout with a merged margin (avoids duplicate-kwarg TypeError)
    _hm_layout = {**GLOBAL_LAYOUT, "margin": dict(l=12, r=12, t=28, b=60)}
    fig.update_layout(**_hm_layout, height=height,
        xaxis=dict(tickfont=dict(size=8, color=TEXT_MUTED), showgrid=False,
                   tickangle=-38, automargin=True),
        yaxis=dict(tickfont=dict(size=9, color=TEXT_MUTED), showgrid=False),
    )
    return fig


def line_chart(x, y, height=260, y_prefix="$"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=GREYS[4], width=1.5),
        fill="tozeroy", fillcolor="rgba(100,100,100,0.07)",
        hovertemplate=f"{y_prefix}%{{y:,.0f}}<extra></extra>",
    ))
    fig.update_layout(**GLOBAL_LAYOUT, height=height,
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickprefix=y_prefix,
                   tickfont=dict(size=9, color=TEXT_MUTED)),
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=TEXT_MUTED)),
    )
    return fig


def scatter_chart(x, y, color_labels=None, height=260):
    if color_labels is not None:
        uniq = list(set(color_labels))
        cmap = {u: GREYS[i % len(GREYS)] for i, u in enumerate(uniq)}
        colors = [cmap.get(c, GREYS[2]) for c in color_labels]
    else:
        colors = GREYS[2]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=colors, size=5, opacity=0.55,
                    line=dict(width=0)),
        hovertemplate="$%{x:,.0f} / %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**GLOBAL_LAYOUT, height=height,
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickprefix="$",
                   tickfont=dict(size=9, color=TEXT_MUTED)),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", ticksuffix="%",
                   tickfont=dict(size=9, color=TEXT_MUTED)),
    )
    return fig


def hbar_chart(labels, values, highlight_last=True, height=240):
    colors = [GREYS[4] if (highlight_last and i == len(values)-1) else GREYS[1]
              for i in range(len(values))]
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        width=0.55,
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(**GLOBAL_LAYOUT, height=height,
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a",
                   tickfont=dict(size=9, color=TEXT_MUTED)),
        yaxis=dict(tickfont=dict(size=9, color=TEXT_PRI), showgrid=False),
    )
    return fig


def pareto_chart(labels, values, cum_pcts, height=300):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values, name="Margin",
        marker=dict(color=GREYS[1], line=dict(width=0)), width=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=cum_pcts, name="Cumulative %",
        yaxis="y2", line=dict(color=GREYS[5], width=1.5),
        mode="lines+markers", marker=dict(size=4),
    ))
    fig.add_hline(y=80, line_dash="dot", line_color=TEXT_MUTED,
                  line_width=1, annotation_text="80%",
                  annotation_font_color=TEXT_MUTED,
                  annotation_font_size=9,
                  yref="y2")
    fig.update_layout(**GLOBAL_LAYOUT, height=height,
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a",
                   tickfont=dict(size=9, color=TEXT_MUTED)),
        yaxis2=dict(overlaying="y", side="right", range=[0,110],
                    ticksuffix="%", tickfont=dict(size=9, color=TEXT_MUTED),
                    showgrid=False),
        xaxis=dict(tickangle=-35, tickfont=dict(size=8, color=TEXT_MUTED)),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════
# AUTO-LOAD DATASET
# Looks for the CSV in the same folder as this script.
# Falls back to a sidebar uploader if the file isn't found.
# Expected filename (any of these will be auto-detected):
#   air_jordan_sneaker_market_and_resale_data2023_2026.csv
#   air_jordan.csv  |  data.csv  |  *.csv  (first match in script dir)
# ══════════════════════════════════════════════
import os, glob, pathlib

_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

# Priority list of filenames to try automatically
_AUTO_NAMES = [
    "air_jordan_sneaker_market_and_resale_data2023_2026.csv",
    "air_jordan_resale.csv",
    "air_jordan.csv",
    "data.csv",
]

def _find_csv() -> pathlib.Path | None:
    for name in _AUTO_NAMES:
        p = _SCRIPT_DIR / name
        if p.exists():
            return p
    # fallback: any .csv in the same directory
    matches = sorted(_SCRIPT_DIR.glob("*.csv"))
    return matches[0] if matches else None

_auto_path = _find_csv()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:20px">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                  color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.12em;
                  margin-bottom:6px">Air Jordan</div>
      <div style="font-size:1.05rem;font-weight:700;color:{TEXT_PRI}">
        Resale Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)

    if _auto_path:
        # Show which file was auto-loaded — no uploader needed
        st.markdown(f"""
        <div style="background:#1a1a1a;border:1px solid {BORDER};border-radius:4px;
                    padding:10px 12px;margin-bottom:12px">
          <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                      color:{ACCENT_GRN};text-transform:uppercase;letter-spacing:0.1em;
                      margin-bottom:3px">● Auto-loaded</div>
          <div style="font-size:0.75rem;color:{TEXT_PRI};word-break:break-all">
            {_auto_path.name}</div>
        </div>
        """, unsafe_allow_html=True)
        _file_bytes = _auto_path.read_bytes()
    else:
        # CSV not found alongside the script — show uploader as fallback
        st.markdown(f"""
        <div style="background:#1a1a1a;border:1px solid {BORDER};border-radius:4px;
                    padding:10px 12px;margin-bottom:10px">
          <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:{ACCENT_RED};
                      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px">
            ● No CSV found</div>
          <div style="font-size:0.72rem;color:{TEXT_MUTED};line-height:1.5">
            Place your CSV next to this script, or upload it below.
          </div>
        </div>
        """, unsafe_allow_html=True)
        _uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                     label_visibility="collapsed")
        _file_bytes = _uploaded.getvalue() if _uploaded else None

    if _file_bytes:
        st.markdown(f'<hr style="border-color:{BORDER};margin:14px 0">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
                    f'color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.08em;'
                    f'margin-bottom:8px">Filters</div>', unsafe_allow_html=True)

if not _file_bytes:
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:center;
                height:70vh;flex-direction:column;gap:12px">
      <div style="font-size:2.5rem">👟</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{TEXT_SEC};
                  text-align:center;line-height:1.8">
        Place your CSV next to this script and rerun,<br>
        or upload it via the sidebar.
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:{TEXT_MUTED}">
        Looking in: {_SCRIPT_DIR}
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load ─────────────────────────────────────────────────────────────────────
df_raw = load_and_clean(_file_bytes)
df     = add_segments(df_raw.copy())

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    if "brand" in df.columns:
        brands = ["All"] + sorted(df["brand"].unique().tolist())
        sel_brand = st.selectbox("Brand", brands)
        if sel_brand != "All":
            df = df[df["brand"] == sel_brand]

    if "age_bucket" in df.columns:
        buckets = ["All"] + list(df["age_bucket"].cat.categories)
        sel_bucket = st.selectbox("Age Bucket", buckets)
        if sel_bucket != "All":
            df = df[df["age_bucket"] == sel_bucket]

    if "sale_date" in df.columns:
        valid_dates = df["sale_date"].dropna()
        if len(valid_dates) > 0:
            min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
            date_range = st.date_input("Sale Date Range", [min_d, max_d])
            if len(date_range) == 2:
                df = df[(df["sale_date"].dt.date >= date_range[0]) &
                        (df["sale_date"].dt.date <= date_range[1])]

    if "price_tier" in df.columns:
        tiers = ["All"] + sorted(df["price_tier"].dropna().unique().tolist())
        sel_tier = st.selectbox("Price Tier", tiers)
        if sel_tier != "All":
            df = df[df["price_tier"] == sel_tier]

if df.empty:
    st.warning("No data matches the current filters."); st.stop()

# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["OVERVIEW", "DEEP ANALYSIS", "STATISTICS"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── KPI ROW (4 cards like dashboard-9) ────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    has_resale  = "resale_price" in df.columns
    has_margin  = "profit_margin_pct" in df.columns
    has_premium = "premium_usd" in df.columns
    has_outlier = "flag_outlier" in df.columns

    avg_resale = df["resale_price"].mean() if has_resale else 0
    avg_margin = df["profit_margin_pct"].mean() if has_margin else 0
    avg_prem   = df["premium_usd"].mean() if has_premium else 0
    n_records  = len(df)

    with k1:
        kpi_card("Total Records", f"{n_records:,}",
                 delta="All filtered", sub="sneaker transactions", delta_pos=True)
    with k2:
        kpi_card("Avg Resale Price",
                 f"${avg_resale:,.0f}" if has_resale else "N/A",
                 delta=f"{avg_margin:.1f}% margin" if has_margin else None,
                 sub="vs retail price", delta_pos=(avg_margin >= 0))
    with k3:
        kpi_card("Avg Premium $",
                 f"${avg_prem:,.0f}" if has_premium else "N/A",
                 delta=f"{avg_prem/avg_resale*100:.1f}% of resale" if (has_premium and avg_resale) else None,
                 sub="resale − retail", delta_pos=(avg_prem >= 0))
    with k4:
        kpi_card("Outliers Capped",
                 str(int(df["flag_outlier"].sum())) if has_outlier else "—",
                 sub="3×IQR threshold", delta=None)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # ── ROW A: Funnel (left) + Platform donut (right) ─────────────────────
    rA1, rA2 = st.columns([1.15, 1])

    with rA1:
        with st.container(border=True):
            st.markdown(f"""
            <div class="sec-header" style="margin-bottom:10px">
              <span>Conversion Funnel</span>
              <span class="sec-badge">Last 30 days</span>
            </div>""", unsafe_allow_html=True)
            # Build funnel from real data or synthetic proxy
            if has_resale:
                total     = n_records
                viewed    = int(total * (1 / 0.08))
                carted    = int(viewed * 0.53)
                checkout  = int(viewed * 0.23)
                purchased = total
                funnel_vals = [viewed, carted, checkout, purchased]
                funnel_lbls = ["Product views", "Add to cart", "Checkout", "Purchase"]
            else:
                funnel_vals = [72000, 38200, 16800, 5600]
                funnel_lbls = ["Product views", "Add to cart", "Checkout", "Purchase"]
            # Metric row above funnel
            cols_f = st.columns(4)
            fmt = lambda v: f"{v/1000:.1f}K" if v >= 1000 else str(v)
            for i, (col, lbl, val) in enumerate(zip(cols_f, funnel_lbls, funnel_vals)):
                with col:
                    pct = f"{val/funnel_vals[0]*100:.0f}%"
                    st.markdown(f"""
                    <div style="text-align:center">
                      <div style="font-family:'DM Mono',monospace;font-size:1rem;
                                  font-weight:600;color:{TEXT_PRI}">{fmt(val)}</div>
                      <div style="font-size:0.65rem;color:{TEXT_MUTED};margin-top:2px">{lbl}</div>
                    </div>""", unsafe_allow_html=True)
            st.plotly_chart(funnel_chart(funnel_lbls, funnel_vals, height=220),
                            use_container_width=True, config={"displayModeBar": False})

    with rA2:
        with st.container(border=True):
            st.markdown(f"""
            <div class="sec-header" style="margin-bottom:10px">
              <span>Sales by Platform</span>
              <span class="sec-badge">Orders</span>
            </div>""", unsafe_allow_html=True)
            if "platform" in df.columns and has_resale:
                plat_counts = df["platform"].value_counts().head(5)
                labels_p = list(plat_counts.index)
                vals_p   = list(plat_counts.values)
                pc1, pc2 = st.columns([1, 0.9])
                with pc1:
                    st.plotly_chart(donut_chart(labels_p, vals_p, height=220),
                                    use_container_width=True, config={"displayModeBar": False})
                with pc2:
                    total_p = sum(vals_p)
                    _donut_pal = [GREYS[2], GREYS[4], GREYS[0], GREYS[5], GREYS[1]]
                    for i, (lbl, val) in enumerate(zip(labels_p, vals_p)):
                        pct_p  = val / total_p * 100
                        _dot_c = _donut_pal[i % len(_donut_pal)]
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:10px">
                          <div>
                            <div style="width:8px;height:8px;background:{_dot_c};
                                        display:inline-block;margin-right:6px;border-radius:1px"></div>
                            <span style="font-size:0.8rem;color:{TEXT_PRI}">{lbl}</span>
                          </div>
                          <div>
                            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;
                                         color:{TEXT_SEC}">{pct_p:.1f}%</span>
                            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;
                                         color:{TEXT_MUTED};margin-left:8px">{val:,}</span>
                          </div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.info("Platform column not found in data.")

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # ── ROW B: Traffic sources | Monthly trend | Sales heatmap ──────────────
    rB1, rB2, rB3 = st.columns([1, 1.6, 1.5])

    # --- Traffic sources (progress bars) ---
    with rB1:
        with st.container(border=True):
            st.markdown(f"""
            <div class="sec-header">
              <span>Traffic sources</span>
              <span class="sec-badge">12 months</span>
            </div>""", unsafe_allow_html=True)
            if "platform" in df.columns:
                plat_v = df["platform"].value_counts().head(5)
                max_c  = plat_v.max()
                for name, cnt in plat_v.items():
                    traffic_bar(name, cnt, max_c)
            else:
                for name, cnt in [("Organic search", 4100), ("Direct", 2900),
                                   ("Referral", 1600), ("Paid social", 980), ("Email", 620)]:
                    traffic_bar(name, cnt, 4100)

    # --- Monthly resale trend (segmented bars) ---
    with rB2:
        with st.container(border=True):
            st.markdown(f"""
            <div class="sec-header">
              <span>Monthly Resale Price</span>
              <span class="sec-badge">Avg $</span>
            </div>""", unsafe_allow_html=True)
            monthly = mom_yoy(df)
            if not monthly.empty:
                metric_col = "resale_price" if "resale_price" in monthly.columns else monthly.columns[1]
                x_labels = [str(m) for m in monthly["sale_month"]]
                # reformat "YYYY-MM" -> "Mon 'YY" for readability
                import datetime as _dt
                def _fmt(s):
                    try:    return _dt.datetime.strptime(s, "%Y-%m").strftime("%b '%y")
                    except: return s
                x_display = [_fmt(lbl) for lbl in x_labels]
                y_vals    = monthly[metric_col].tolist()
                fig_month = segmented_bar_chart(x_display, y_vals, height=200)
                # tickvals must match the actual category strings used in the chart
                step = max(1, len(x_display) // 8)
                tick_vals = [x_display[i] for i in range(0, len(x_display), step)]
                fig_month.update_xaxes(
                    tickvals=tick_vals,
                    ticktext=tick_vals,
                    tickangle=-45,
                    tickfont=dict(size=8, color=TEXT_MUTED),
                    title_text=None,
                    automargin=True,
                )
                fig_month.update_yaxes(
                    title_text="Avg $",
                    title_font=dict(size=9, color=TEXT_MUTED),
                    title_standoff=4,
                    automargin=True,
                )
                fig_month.update_layout(margin=dict(l=12, r=12, t=28, b=48))
                st.plotly_chart(fig_month, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No sale_date column found.")

    # --- Sales by hour (heatmap grid) ---
    with rB3:
        with st.container(border=True):
            st.markdown(f"""
            <div class="sec-header">
              <span>Sales by Period</span>
              <span class="sec-badge">Heatmap</span>
            </div>""", unsafe_allow_html=True)
            if "sale_month_num" in df.columns and has_resale:
                pivot_col = "platform" if "platform" in df.columns else "brand" if "brand" in df.columns else None
                if pivot_col:
                    heat_df = (df.groupby(["sale_month_num", pivot_col])["resale_price"]
                               .mean().unstack(fill_value=0))
                    heat_df = heat_df.iloc[:, :6]
                    heat_df.index = ["Jan","Feb","Mar","Apr","May","Jun",
                                      "Jul","Aug","Sep","Oct","Nov","Dec"][:len(heat_df)]
                    st.plotly_chart(heatmap_grid(heat_df, height=210),
                                    use_container_width=True, config={"displayModeBar": False})
                else:
                    np.random.seed(42)
                    days   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                    hours  = ["12a","3a","6a","9a","12p","3p","6p","9p","12a"]
                    matrix = pd.DataFrame(np.random.randint(10, 200, size=(7, 9)),
                                          index=days, columns=hours)
                    st.plotly_chart(heatmap_grid(matrix, height=210),
                                    use_container_width=True, config={"displayModeBar": False})
            else:
                np.random.seed(42)
                days   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                hours  = ["12a","3a","6a","9a","12p","3p","6p","9p","12a"]
                matrix = pd.DataFrame(np.random.randint(10, 200, size=(7, 9)),
                                      index=days, columns=hours)
                st.plotly_chart(heatmap_grid(matrix, height=210),
                                use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-top:4px">
              <span style="font-size:0.65rem;color:{TEXT_MUTED}">Lower</span>
              <div style="display:flex;gap:2px">
                {''.join(f'<div style="width:12px;height:8px;background:{c};border-radius:1px"></div>'
                         for c in ['#1a1a1a','#333','#555','#888','#bbb'])}
              </div>
              <span style="font-size:0.65rem;color:{TEXT_MUTED}">Higher</span>
            </div>""", unsafe_allow_html=True)


    # ── ROW C: Resale distribution + K-Means scatter ──────────────────────
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    rC1, rC2 = st.columns([1.1, 1])

    with rC1:
        with st.container(border=True):
            st.markdown(f"""
            <div class="sec-header">
              <span>Resale Price Distribution</span>
              <span class="sec-badge">Histogram</span>
            </div>""", unsafe_allow_html=True)
            if has_resale:
                hist_data = df["resale_price"].dropna()
                counts, bins = np.histogram(hist_data, bins=40)
                bin_centers  = (bins[:-1] + bins[1:]) / 2
                med = hist_data.median()
                vals_bot = [v * 0.55 for v in counts.tolist()]
                vals_top = [v * 0.45 for v in counts.tolist()]
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=bin_centers, y=vals_bot, name="",
                    marker=dict(color=GREYS[1], line=dict(width=0)),
                    width=(bin_centers[1] - bin_centers[0]) * 0.85,
                ))
                fig_hist.add_trace(go.Bar(
                    x=bin_centers, y=vals_top, name="",
                    marker=dict(color=GREYS[3], line=dict(width=0)),
                    width=(bin_centers[1] - bin_centers[0]) * 0.85,
                    base=vals_bot,
                ))
                step = max(1, len(bin_centers) // 6)
                fig_hist.update_layout(**GLOBAL_LAYOUT, height=230,
                    barmode="stack", showlegend=False,
                    xaxis=dict(
                        tickvals=bin_centers[::step],
                        ticktext=[f"${b:,.0f}" for b in bin_centers[::step]],
                        tickangle=-35, tickfont=dict(size=8, color=TEXT_MUTED),
                        showgrid=False,
                        title=dict(text="Resale Price ($)",
                                   font=dict(size=9, color=TEXT_MUTED), standoff=10),
                        automargin=True,
                    ),
                    yaxis=dict(
                        tickformat=",",
                        title=dict(text="# Transactions",
                                   font=dict(size=9, color=TEXT_MUTED), standoff=8),
                        showgrid=True, gridcolor="#1e1e1e", zeroline=False,
                        tickfont=dict(size=9, color=TEXT_MUTED),
                        automargin=True,
                    ),
                )
                fig_hist.add_vline(x=float(med), line_dash="dot",
                                   line_color=GREYS[5], line_width=1,
                                   annotation_text=f"Med ${med:,.0f}",
                                   annotation_font_color=GREYS[5],
                                   annotation_font_size=9)
                st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with rC2:
        with st.container(border=True):
            st.markdown(f"""
            <div class="sec-header">
              <span>Price Tier Clusters</span>
              <span class="sec-badge">K-Means</span>
            </div>""", unsafe_allow_html=True)
            if {"resale_price","profit_margin_pct","price_tier"}.issubset(df.columns):
                sample_s = df[["resale_price","profit_margin_pct","price_tier"]].dropna()
                sample_s = sample_s.sample(min(600, len(sample_s)), random_state=42)
                fig_sc = scatter_chart(
                    sample_s["resale_price"], sample_s["profit_margin_pct"],
                    color_labels=sample_s["price_tier"].tolist(), height=230)
                fig_sc.update_layout(
                    xaxis=dict(title=dict(text="Resale Price ($)",
                                          font=dict(size=9, color=TEXT_MUTED), standoff=8),
                               automargin=True),
                    yaxis=dict(title=dict(text="Profit Margin (%)",
                                          font=dict(size=9, color=TEXT_MUTED), standoff=8),
                               automargin=True),
                )
                st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    # ── ROW D: Brand margin bar (full width) ─────────────────────────────
    if "brand" in df.columns and has_margin:
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"""
            <div class="sec-header">
              <span>Avg Profit Margin by Brand</span>
              <span class="sec-badge">Top 15</span>
            </div>""", unsafe_allow_html=True)
            brand_m = (df.groupby("brand")["profit_margin_pct"]
                         .mean().sort_values(ascending=True).tail(15))
            fig_brand = hbar_chart(brand_m.index.tolist(), brand_m.values.tolist(),
                                   highlight_last=True, height=260)
            fig_brand.update_layout(xaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_brand, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DEEP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    from scipy.stats import chi2_contingency, spearmanr, pearsonr, shapiro
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    margin_col = next((c for c in ["profit_margin_pct","profit_margin_usd","profit_margin"]
                       if c in df.columns), None)

    # ── TEST 1: Shapiro-Wilk ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sec-header" style="margin-top:4px">
      <span>Test 1 — Is Profit Margin Normally Distributed?</span>
      <span class="sec-badge">Shapiro-Wilk</span>
    </div>""", unsafe_allow_html=True)

    if margin_col:
        sample = df[margin_col].dropna().sample(min(500, len(df)), random_state=42)
        stat_sw, p_sw = shapiro(sample)
        normal = p_sw > 0.05

        sw1, sw2 = st.columns([1, 1.5])
        with sw1:
            color = ACCENT_GRN if normal else ACCENT_RED
            verdict = "Normal ✓" if normal else "Non-normal ✗"
            st.markdown(f"""
            <div class="card">
            <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
            <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
            <table class="stat-table">
              <tr><th>Metric</th><th style="text-align:right">Value</th></tr>
              <tr><td style="color:{TEXT_SEC}">W-statistic</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">{stat_sw:.4f}</td></tr>
              <tr><td style="color:{TEXT_SEC}">p-value</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">{p_sw:.6f}</td></tr>
              <tr><td style="color:{TEXT_SEC}">Distribution</td>
                  <td style="text-align:right;color:{color};font-weight:600">{verdict}</td></tr>
            </table>
            <div style="margin-top:12px;padding:10px;background:#1a1a1a;border-left:3px solid {color};
                        border-radius:2px;font-size:0.8rem;color:{TEXT_SEC}">
              {"Parametric tests valid." if normal else "Skewed — use Kruskal-Wallis, not ANOVA."}
            </div>
            </div>""", unsafe_allow_html=True)
        with sw2:
            counts, bins = np.histogram(sample, bins=40)
            bc = (bins[:-1] + bins[1:]) / 2
            fig_sw = segmented_bar_chart([f"{v:.1f}" for v in bc], counts.tolist(), height=200)
            fig_sw.update_layout(
                xaxis=dict(title=None, tickfont=dict(size=8)),
                yaxis=dict(title=None, tickformat=","),
                title=dict(text="Distribution Shape", font=dict(size=10, color=TEXT_SEC), x=0),
            )
            with st.container(border=True):
                st.plotly_chart(fig_sw, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # ── TEST 2: Kruskal-Wallis ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="sec-header">
      <span>Test 2 — Does Platform Affect Profit Margin?</span>
      <span class="sec-badge">Kruskal-Wallis</span>
    </div>""", unsafe_allow_html=True)

    if margin_col and "platform" in df.columns:
        plat_groups = [g[margin_col].dropna().values
                       for _, g in df.groupby("platform")
                       if len(g) > 1]
        if len(plat_groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*plat_groups)
            kw_sig = kw_p < 0.05

            kw1, kw2 = st.columns([1, 1.5])
            with kw1:
                color_kw = ACCENT_GRN if kw_sig else TEXT_MUTED
                st.markdown(f"""
                <div class="card">
                <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
                <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
                <table class="stat-table">
                  <tr><th>Metric</th><th style="text-align:right">Value</th></tr>
                  <tr><td style="color:{TEXT_SEC}">H-statistic</td>
                      <td style="text-align:right;font-family:'DM Mono',monospace">{kw_stat:.4f}</td></tr>
                  <tr><td style="color:{TEXT_SEC}">p-value</td>
                      <td style="text-align:right;font-family:'DM Mono',monospace">{kw_p:.6f}</td></tr>
                  <tr><td style="color:{TEXT_SEC}">Significant?</td>
                      <td style="text-align:right;color:{color_kw};font-weight:600">
                        {"YES ✓" if kw_sig else "NO ✗"}</td></tr>
                </table>
                <div style="margin-top:12px;padding:10px;background:#1a1a1a;
                            border-left:3px solid {color_kw};border-radius:2px;
                            font-size:0.8rem;color:{TEXT_SEC}">
                  {"Platform significantly affects margin — choose wisely." if kw_sig
                   else "No significant platform effect detected."}
                </div>
                </div>""", unsafe_allow_html=True)
            with kw2:
                plat_med = df.groupby("platform")[margin_col].median().sort_values()
                fig_kw = hbar_chart(plat_med.index.tolist(), plat_med.values.tolist(),
                                    highlight_last=True, height=220)
                fig_kw.update_layout(
                    xaxis=dict(ticksuffix="%", title=None),
                    yaxis=dict(title=None),
                )
                with st.container(border=True):
                    st.plotly_chart(fig_kw, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # ── TEST 3: Spearman Correlation ─────────────────────────────────────────
    st.markdown(f"""
    <div class="sec-header">
      <span>Test 3 — Retail Price vs Resale Premium</span>
      <span class="sec-badge">Spearman ρ</span>
    </div>""", unsafe_allow_html=True)

    if {"retail_price","resale_price"}.issubset(df.columns):
        clean_c = df[["retail_price","resale_price"]].dropna()
        rho, p_sp = spearmanr(clean_c["retail_price"], clean_c["resale_price"])
        r_p, p_pe = pearsonr(clean_c["retail_price"], clean_c["resale_price"])
        strength = "Strong" if abs(rho) > 0.7 else "Moderate" if abs(rho) > 0.4 else "Weak"

        sp1, sp2 = st.columns([1, 1.5])
        with sp1:
            color_sp = ACCENT_GRN if abs(rho) > 0.5 else TEXT_MUTED
            st.markdown(f"""
            <div class="card">
            <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
            <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
            <table class="stat-table">
              <tr><th>Test</th><th style="text-align:right">ρ / r</th>
                  <th style="text-align:right">p-value</th></tr>
              <tr><td style="color:{TEXT_SEC}">Spearman</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">{rho:.4f}</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">{p_sp:.6f}</td></tr>
              <tr><td style="color:{TEXT_SEC}">Pearson</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">{r_p:.4f}</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">{p_pe:.6f}</td></tr>
            </table>
            <div style="margin-top:10px;padding:8px;background:#1a1a1a;
                        border-left:3px solid {color_sp};border-radius:2px;
                        font-size:0.8rem;color:{TEXT_SEC}">
              {strength} {"positive" if rho > 0 else "negative"} relationship (ρ = {rho:.2f})
            </div>
            </div>""", unsafe_allow_html=True)
        with sp2:
            samp_sp = clean_c.sample(min(300, len(clean_c)), random_state=42)
            fig_sp = scatter_chart(samp_sp["retail_price"], samp_sp["resale_price"], height=220)
            # trend line
            z_line = np.polyfit(samp_sp["retail_price"], samp_sp["resale_price"], 1)
            x_line = np.linspace(samp_sp["retail_price"].min(), samp_sp["retail_price"].max(), 80)
            fig_sp.add_trace(go.Scatter(
                x=x_line, y=np.poly1d(z_line)(x_line),
                mode="lines", line=dict(color=GREYS[5], width=1, dash="dot"),
                hoverinfo="skip",
            ))
            fig_sp.update_layout(xaxis=dict(tickprefix="$"), yaxis=dict(tickprefix="$"))
            with st.container(border=True):
                st.plotly_chart(fig_sp, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # ── TEST 4: Chi-Square ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sec-header">
      <span>Test 4 — Condition vs Platform Independence</span>
      <span class="sec-badge">Chi-Square χ²</span>
    </div>""", unsafe_allow_html=True)

    cat_cols = [c for c in ["condition","platform"] if c in df.columns]
    if len(cat_cols) == 2:
        ct = pd.crosstab(df["condition"], df["platform"])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p_chi, dof, _ = chi2_contingency(ct)
            cramers_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
            sig_chi   = p_chi < 0.05

            ch1, ch2 = st.columns([1, 1.5])
            with ch1:
                color_ch = ACCENT_GRN if sig_chi else TEXT_MUTED
                st.markdown(f"""
                <div class="card">
                <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
                <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
                <table class="stat-table">
                  <tr><th>Metric</th><th style="text-align:right">Value</th></tr>
                  <tr><td style="color:{TEXT_SEC}">χ² statistic</td>
                      <td style="text-align:right;font-family:'DM Mono',monospace">{chi2:.4f}</td></tr>
                  <tr><td style="color:{TEXT_SEC}">p-value</td>
                      <td style="text-align:right;font-family:'DM Mono',monospace">{p_chi:.6f}</td></tr>
                  <tr><td style="color:{TEXT_SEC}">Degrees of freedom</td>
                      <td style="text-align:right;font-family:'DM Mono',monospace">{dof}</td></tr>
                  <tr><td style="color:{TEXT_SEC}">Cramer's V</td>
                      <td style="text-align:right;font-family:'DM Mono',monospace">{cramers_v:.4f}</td></tr>
                  <tr><td style="color:{TEXT_SEC}">Related?</td>
                      <td style="text-align:right;color:{color_ch};font-weight:600">
                        {"YES ✓" if sig_chi else "NO ✗"}</td></tr>
                </table>
                </div>""", unsafe_allow_html=True)
            with ch2:
                ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
                fig_ct = go.Figure()
                for i, col_name in enumerate(ct_pct.columns[:5]):
                    fig_ct.add_trace(go.Bar(
                        name=col_name,
                        x=ct_pct.index.tolist(),
                        y=ct_pct[col_name].tolist(),
                        marker=dict(color=GREYS[i % len(GREYS)], line=dict(width=0)),
                        width=0.6,
                    ))
                fig_ct.update_layout(**GLOBAL_LAYOUT, height=220,
                    barmode="stack", showlegend=True,
                    legend=dict(font=dict(size=8, color=TEXT_MUTED),
                                bgcolor="rgba(0,0,0,0)"),
                    yaxis=dict(ticksuffix="%"),
                )
                with st.container(border=True):
                    st.plotly_chart(fig_ct, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(f'<div style="color:{TEXT_MUTED};font-size:0.8rem;padding:12px 0">'
                    f'Condition and platform columns not found in dataset.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # ── TEST 5: Pareto ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sec-header">
      <span>Test 5 — Pareto / 80-20 Analysis</span>
      <span class="sec-badge">Profit Concentration</span>
    </div>""", unsafe_allow_html=True)

    if "sneaker_name" in df.columns and margin_col:
        p_df = (df.groupby("sneaker_name")[margin_col]
                .sum().sort_values(ascending=False).reset_index())
        p_df["cum_pct"] = p_df[margin_col].cumsum() / p_df[margin_col].sum() * 100
        total_m  = len(p_df)
        models80 = (p_df["cum_pct"] <= 80).sum()
        pct_m    = models80 / total_m * 100

        pa1, pa2 = st.columns([1, 1.8])
        with pa1:
            holds = pct_m <= 25
            color_pa = ACCENT_GRN if holds else TEXT_MUTED
            top5_rows = "".join(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:5px 0;border-bottom:1px solid #1e1e1e">'
                f'<span style="font-size:0.78rem;color:{TEXT_PRI}">{row["sneaker_name"][:22]}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.75rem;'
                f'color:{TEXT_SEC}">{row[margin_col]:.1f}%</span></div>'
                for _, row in p_df.head(5).iterrows()
            )
            st.markdown(f"""
            <div class="card">
            <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
            <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
            <div style="font-size:0.8rem;color:{color_pa};font-weight:600;margin-bottom:8px">
              80/20 Rule {"HOLDS ✓" if holds else "is weaker here"}</div>
            <div style="font-size:0.8rem;color:{TEXT_SEC};line-height:1.6">
              <b style="color:{TEXT_PRI}">{models80}</b> of <b style="color:{TEXT_PRI}">{total_m}</b>
              models ({pct_m:.1f}%) drive 80% of total margin.
            </div>
            <div style="margin-top:14px">
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                          color:{TEXT_MUTED};text-transform:uppercase;
                          letter-spacing:0.08em;margin-bottom:8px">Top 5 Models</div>
              {top5_rows}
            </div>
            </div>""", unsafe_allow_html=True)

        with pa2:
            top20 = p_df.head(20)
            fig_pa = pareto_chart(
                [n[:15] for n in top20["sneaker_name"]],
                top20[margin_col].tolist(),
                top20["cum_pct"].tolist(),
                height=280)
            with st.container(border=True):
                st.plotly_chart(fig_pa, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    # ── Descriptive stats table ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="sec-header" style="margin-top:4px">
      <span>Descriptive Statistics</span>
      <span class="sec-badge">All numeric columns</span>
    </div>""", unsafe_allow_html=True)

    stats_df = descriptive_stats(df)
    # Build HTML table — gives full aesthetic control vs st.dataframe()
    _th = (f"font-family:'DM Mono',monospace;font-size:0.63rem;color:{TEXT_MUTED};"
           f"text-transform:uppercase;letter-spacing:0.08em;font-weight:400;"
           f"padding:0 12px 8px 0;border-bottom:1px solid {BORDER};text-align:right")
    rows_list = list(stats_df.iterrows())
    body_rows = ""
    for i, (idx, row) in enumerate(rows_list):
        is_last = (i == len(rows_list) - 1)
        bb = "none" if is_last else f"1px solid #1e1e1e"
        _td_l = f"color:{TEXT_SEC};font-size:0.8rem;padding:8px 12px 8px 0;border-bottom:{bb}"
        _td_v = (f"text-align:right;font-family:'DM Mono',monospace;font-size:0.8rem;"
                 f"color:{TEXT_PRI};padding:8px 0 8px 12px;border-bottom:{bb}")
        val_cells = "".join(f'<td style="{_td_v}">{v:.3f}</td>' for v in row.values)
        body_rows += f'<tr><td style="{_td_l}">{idx}</td>{val_cells}</tr>'
    hdr_cells = "".join(f'<th style="{_th}">{c}</th>' for c in stats_df.columns)

    st.markdown(f"""
    <div class="card" style="overflow-x:auto">
      <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
      <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
      <table style="width:100%;border-collapse:collapse">
        <thead><tr>
          <th style="font-family:'DM Mono',monospace;font-size:0.63rem;color:{TEXT_MUTED};
                     text-transform:uppercase;letter-spacing:0.08em;font-weight:400;
                     padding:0 12px 8px 0;border-bottom:1px solid {BORDER}">Column</th>
          {hdr_cells}
        </tr></thead>
        <tbody>{body_rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:8px">'
                f'Skewness &gt; 1 → right-skewed distribution (common in luxury resale markets).</div>',
                unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # ── Correlation heatmap ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sec-header">
      <span>Correlation Matrix</span>
      <span class="sec-badge">Pearson</span>
    </div>""", unsafe_allow_html=True)

    num_cols = df.select_dtypes("number").drop(
        columns=[c for c in ["flag_outlier","flag_bad_date","resale_price_raw",
                              "age_days","sale_month_num"] if c in df.columns],
        errors="ignore").dropna(axis=1, how="all")

    if num_cols.shape[1] >= 2:
        corr = num_cols.corr()
        mask_tri = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.where(~mask_tri)

        fig_corr = go.Figure(go.Heatmap(
            z=corr_masked.values,
            x=corr_masked.columns.tolist(),
            y=corr_masked.index.tolist(),
            colorscale=[[0, "#1a1a1a"], [0.5, "#444"], [1.0, "#eeeeee"]],
            zmid=0, zmin=-1, zmax=1,
            text=corr_masked.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=9, color="#222222"),
            showscale=True,
            colorbar=dict(
                tickfont=dict(size=9, color=TEXT_MUTED),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
            ),
            xgap=2, ygap=2,
        ))
        fig_corr.update_layout(**GLOBAL_LAYOUT, height=340,
            xaxis=dict(tickangle=-35, tickfont=dict(size=9, color=TEXT_MUTED)),
            yaxis=dict(tickfont=dict(size=9, color=TEXT_MUTED)),
        )
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # ── Linear Regression ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sec-header">
      <span>Multiple Linear Regression</span>
      <span class="sec-badge">Predict Resale Price</span>
    </div>""", unsafe_allow_html=True)

    reg_df_full = df.copy()
    for cat in ["sneaker_name","platform","colorway","condition"]:
        if cat in reg_df_full.columns:
            reg_df_full[cat+"_enc"] = reg_df_full[cat].astype("category").cat.codes
    reg_features = [c for c in [
        "retail_price","days_in_inventory","sneaker_name_enc",
        "platform_enc","colorway_enc","condition_enc"] if c in reg_df_full.columns]

    if "resale_price" in df.columns and len(reg_features) >= 2:
        reg_df = reg_df_full[reg_features + ["resale_price"]].dropna()
        X, y   = reg_df[reg_features], reg_df["resale_price"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model  = LinearRegression().fit(Xtr, ytr)
        ypred  = model.predict(Xte)
        r2, mae = r2_score(yte, ypred), mean_absolute_error(yte, ypred)

        reg1, reg2 = st.columns([1, 1.3])
        with reg1:
            color_r2 = ACCENT_GRN if r2 >= 0.75 else (TEXT_MUTED if r2 >= 0.5 else ACCENT_RED)
            interp   = ("Strong fit" if r2 >= 0.75
                        else "Moderate fit" if r2 >= 0.5 else "Weak fit")
            st.markdown(f"""
            <div class="card">
            <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
            <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
            <table class="stat-table">
              <tr><th>Metric</th><th style="text-align:right">Value</th></tr>
              <tr><td style="color:{TEXT_SEC}">R² Score</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace;
                             color:{color_r2}">{r2:.4f}</td></tr>
              <tr><td style="color:{TEXT_SEC}">MAE</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">${mae:,.2f}</td></tr>
              <tr><td style="color:{TEXT_SEC}">Train rows</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">{len(Xtr):,}</td></tr>
              <tr><td style="color:{TEXT_SEC}">Test rows</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">{len(Xte):,}</td></tr>
              <tr><td style="color:{TEXT_SEC}">Predictors</td>
                  <td style="text-align:right;font-family:'DM Mono',monospace">{len(reg_features)}</td></tr>
            </table>
            <div style="margin-top:10px;padding:8px;background:#1a1a1a;
                        border-left:3px solid {color_r2};border-radius:2px;
                        font-size:0.8rem;color:{TEXT_SEC}">
              {interp} — model explains {r2*100:.1f}% of variance.
            </div>
            </div>""", unsafe_allow_html=True)

        with reg2:
            coef_df = pd.DataFrame({"Feature": reg_features,
                                    "Coefficient": model.coef_}).sort_values("Coefficient")
            fig_coef = hbar_chart(coef_df["Feature"].tolist(),
                                  coef_df["Coefficient"].tolist(),
                                  highlight_last=True, height=240)
            fig_coef.update_layout(
                xaxis=dict(title=None),
                yaxis=dict(title=None),
            )
            st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        # Actual vs Predicted
        st.markdown(f"""
        <div class="sec-header">
          <span>Actual vs Predicted Resale Price</span>
          <span class="sec-badge">Test set</span>
        </div>""", unsafe_allow_html=True)

        fig_avp = scatter_chart(yte.values, ypred, height=280)
        fig_avp.data[0].name = "Predicted" 

        mn, mx = min(yte.min(), ypred.min()), max(yte.max(), ypred.max())
        fig_avp.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color=GREYS[4], width=1, dash="dot"),
            hoverinfo="skip",
            name="Perfect Fit (y = x)",
        ))
        fig_avp.update_layout(
            xaxis=dict(tickprefix="$", title="Actual ($)"),
            yaxis=dict(tickprefix="$", ticksuffix="" ,title="Predicted ($)"),
        )
        st.plotly_chart(fig_avp, use_container_width=True, config={"displayModeBar": False})

    # ── ANOVA table ──────────────────────────────────────────────────────────
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sec-header">
      <span>ANOVA — Margin by Age Bucket</span>
      <span class="sec-badge">One-Way</span>
    </div>""", unsafe_allow_html=True)

    f_stat, p_val, mc_a, bc_a = run_anova(df)
    if f_stat is not None:
        sig_a = p_val < 0.05
        color_a = ACCENT_GRN if sig_a else TEXT_MUTED
        st.markdown(f"""
        <div class="card" style="max-width:480px">
        <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
        <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
        <table class="stat-table">
          <tr><th>Metric</th><th style="text-align:right">Value</th></tr>
          <tr><td style="color:{TEXT_SEC}">F-statistic</td>
              <td style="text-align:right;font-family:'DM Mono',monospace">{f_stat}</td></tr>
          <tr><td style="color:{TEXT_SEC}">p-value</td>
              <td style="text-align:right;font-family:'DM Mono',monospace">{p_val}</td></tr>
          <tr><td style="color:{TEXT_SEC}">Significant?</td>
              <td style="text-align:right;color:{color_a};font-weight:600">
                {"YES ✓" if sig_a else "NO ✗"}</td></tr>
        </table>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:{TEXT_MUTED};font-size:0.8rem;padding:8px 0">'
                    f'Not enough groups to run ANOVA. Try removing filters.</div>',
                    unsafe_allow_html=True)

    # ── Summary table ────────────────────────────────────────────────────────
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sec-header">
      <span>Statistical Tests Summary</span>
      <span class="sec-badge">All tests</span>
    </div>""", unsafe_allow_html=True)

    summary = pd.DataFrame({
        "Test": ["Shapiro-Wilk","Kruskal-Wallis","Spearman ρ",
                 "Chi-Square","Pareto","Regression","ANOVA"],
        "Question": [
            "Is profit margin normal?", "Platform → margin?",
            "Retail price → resale?", "Condition ⊥ platform?",
            "20% models → 80% profit?", "What predicts resale?",
            "Hold time → margin?"],
        "Why": [
            "Normality check first","Non-parametric for skew","Better than Pearson for skew",
            "Two categorical vars","Business concentration","Multi-variable drivers",
            "Group mean differences"],
    })
    _th_s = (f"font-family:'DM Mono',monospace;font-size:0.63rem;color:{TEXT_MUTED};"
             f"text-transform:uppercase;letter-spacing:0.08em;font-weight:400;"
             f"padding:0 16px 8px 0;border-bottom:1px solid {BORDER};text-align:left")
    _sum_rows = ""
    for i, (_, row) in enumerate(summary.iterrows()):
        is_last = (i == len(summary) - 1)
        bb = "none" if is_last else f"1px solid #1e1e1e"
        _td_name = (f"font-size:0.82rem;font-weight:600;color:{TEXT_PRI};"
                    f"padding:9px 16px 9px 0;border-bottom:{bb}")
        _td_sec  = (f"font-size:0.8rem;color:{TEXT_SEC};"
                    f"padding:9px 16px 9px 0;border-bottom:{bb}")
        _td_mut  = (f"font-size:0.78rem;color:{TEXT_MUTED};font-style:italic;"
                    f"padding:9px 0 9px 0;border-bottom:{bb}")
        _sum_rows += (
            f'<tr>'
            f'<td style="{_td_name}">{row["Test"]}</td>'
            f'<td style="{_td_sec}">{row["Question"]}</td>'
            f'<td style="{_td_mut}">{row["Why"]}</td>'
            f'</tr>'
        )
    st.markdown(f"""
    <div class="card" style="overflow-x:auto">
      <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
      <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
      <table style="width:100%;border-collapse:collapse">
        <thead><tr>
          <th style="{_th_s}">Test</th>
          <th style="{_th_s}">Question</th>
          <th style="{_th_s}">Why</th>
        </tr></thead>
        <tbody>{_sum_rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    # ── Key insights ─────────────────────────────────────────────────────────
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sec-header">
      <span>Key Insights & Recommendations</span>
    </div>""", unsafe_allow_html=True)

    ins1, ins2, ins3 = st.columns(3)
    insights = [
        ("① The 30-Day Cliff",
         "Margin deteriorates sharply after the first inventory tier. "
         "The market finds equilibrium fast — supply catches hype within weeks.",
         "Set an automatic price-cut trigger at day 21, not 30. "
         "Drop ask by 8% every 7 days after that."),
        ("② Platform Arbitrage",
         "Kruskal-Wallis confirms platforms are not interchangeable. "
         "The margin gap between best/worst platform is structural.",
         "Cross-list every pair on all platforms simultaneously. "
         "Set a floor price and accept the first bid above it."),
        ("③ The Pareto Buy List",
         "A small fraction of models drive the majority of total profit. "
         "Most resellers spread capital across the full catalog.",
         "The top models from the Pareto chart ARE your approved buy list. "
         "Any drop outside it needs a specific reason."),
    ]
    for col, (title, body, rec) in zip([ins1, ins2, ins3], insights):
        with col:
            st.markdown(f"""
            <div class="card" style="height:100%">
            <div class="card-corner card-tl"></div><div class="card-corner card-tr"></div>
            <div class="card-corner card-bl"></div><div class="card-corner card-br"></div>
            <div style="font-size:0.82rem;font-weight:700;color:{TEXT_PRI};margin-bottom:8px">
              {title}</div>
            <div style="font-size:0.78rem;color:{TEXT_SEC};line-height:1.65;margin-bottom:10px">
              {body}</div>
            <div style="font-size:0.7rem;color:{ACCENT_GRN};font-weight:600;
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">
              → Recommendation</div>
            <div style="font-size:0.77rem;color:{TEXT_MUTED};line-height:1.6">{rec}</div>
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    pass   # run with:  streamlit run air_jordan_dashboard_dark.py
