from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import linregress

from utils import add_features, base100, iqr_outliers, load_data, style_axis, yoy

st.set_page_config(page_title="Energy Transition Dashboard", layout="wide")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D9DCE1",
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
        "axes.labelcolor": "#2E3440",
        "xtick.color": "#4C566A",
        "ytick.color": "#4C566A",
    }
)

COLORS = {
    "gross_con": "#3B82F6",
    "total_wind_power": "#14B8A6",
    "total_renewables": "#10B981",
    "total_fossil": "#EF4444",
    "co2_perk_wh": "#F59E0B",
    "net_exchange": "#6366F1",
    "renewables_share_pct": "#22C55E",
    "wind_share_pct": "#0EA5A4",
    "fossil_share_pct": "#F97316",
}


DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data/processed/processed_energy.parquet"
)


@st.cache_data
def load_cached_data():
    return load_data(DATA_PATH)


# Sidebar
st.sidebar.title("Settings")
try:
    df_raw = load_cached_data()
    df = add_features(df_raw)
    df = df[df.index > pd.Timestamp("2014-12-31", tz="UTC")]
except Exception as e:
    st.error(f"Data loading/feature error: {e}")
    st.stop()

# Date filter
min_d, max_d = df.index.min() + pd.Timedelta(days=1), df.index.max()
date_range = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start = pd.Timestamp(date_range[0], tz="UTC")
    end = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
    df_f = df.loc[(df.index >= start) & (df.index < end)]
else:
    df_f = df

st.sidebar.caption("Note: net_exchange > 0 = imports, net_exchange < 0 = exports.")

sample_n = st.sidebar.slider("Scatter sample size", 2000, 80000, 30000, step=2000)

st.title("⚡ Energy Transition Dashboard")
st.caption(
    "Focus: structural transition (wind ↑, fossil ↓, CO₂ ↓), plus seasonality, drivers and extremes."
)

yearly = (
    df_f.groupby("year")[
        [
            "gross_con",
            "total_wind_power",
            "total_renewables",
            "total_fossil",
            "co2_perk_wh",
            "net_exchange",
            "renewables_share_pct",
            "wind_share_pct",
            "fossil_share_pct",
        ]
    ]
    .mean()
    .reset_index()
)

latest_year = int(yearly["year"].max()) if len(yearly) else None

# KPIs
st.subheader("Executive Overview")

row1_cols = st.columns(3)
row2_cols = st.columns(3)

# Latest year KPIs (or overall mean if only one year)
kpi_specs = [
    ("Renewables share (mean %)", "renewables_share_pct", "{:.1f}", True),
    ("Wind share (mean %)", "wind_share_pct", "{:.1f}", True),
    ("CO₂ intensity (g/kWh)", "co2_perk_wh", "{:.0f}", True),
    ("Wind generation (MW)", "total_wind_power", "{:,.0f}", True),
    ("Fossil generation (MW)", "total_fossil", "{:,.0f}", True),
    ("Consumption (MW)", "gross_con", "{:,.0f}", True),
]
kpi_groups = [kpi_specs[:3], kpi_specs[3:]]


def format_value(x, fmt):
    return fmt.format(x) if pd.notna(x) else "—"


def format_yoy(yoy_val):
    return f"{yoy_val:+.1f}% YoY" if pd.notna(yoy_val) else "—"


if latest_year is not None and len(yearly) >= 2:
    y = yearly.set_index("year")

    for col_box, (label, col, fmt, show_yoy) in zip(row1_cols, kpi_specs[:3]):
        last_val = y[col].iloc[-1]
        delta = yoy(y[col]) if show_yoy else np.nan
        col_box.metric(label, format_value(last_val, fmt), format_yoy(delta))

    for col_box, (label, col, fmt, show_yoy) in zip(row2_cols, kpi_specs[3:]):
        last_val = y[col].iloc[-1]
        delta = yoy(y[col]) if show_yoy else np.nan
        col_box.metric(label, format_value(last_val, fmt), format_yoy(delta))
else:
    for col_box, (label, col, fmt, _) in zip(row1_cols, kpi_specs[:3]):
        col_box.metric(label, format_value(df_f[col].mean(), fmt))
    for col_box, (label, col, fmt, _) in zip(row2_cols, kpi_specs[3:]):
        col_box.metric(label, format_value(df_f[col].mean(), fmt))


with st.expander("Definitions"):
    st.write(
        "- Shares are computed relative to gross system consumption (gross_con, incl. transmission losses)."
    )
    st.write("- net_exchange > 0 = imports, net_exchange < 0 = exports.")
    st.write("- Data version: FINAL (up to 31/12/2024).")

# Tabs
tab_overview, tab_season, tab_drivers, tab_extremes, tab_trends = st.tabs(
    ["Transition", "Seasonality", "Drivers", "Extremes", "Trends"]
)

# Transition tab
with tab_overview:
    st.markdown(
        """
        **What this shows:**  
        - Wind generation increases over time while fossil generation declines.  
        - Carbon intensity decreases even as consumption remains stable/increases (evidence of decarbonization).
        """
    )
    st.subheader("Structural Transition (Base 100)")

    labels = {
        "gross_con": "Consumption",
        "total_wind_power": "Wind",
        "total_fossil": "Fossil",
        "co2_perk_wh": "CO2 intensity",
    }
    cols_idx = ["gross_con", "total_wind_power", "total_fossil", "co2_perk_wh"]
    idx_df = yearly.set_index("year")
    idx_base = base100(idx_df, cols_idx)

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in cols_idx:
        ax.plot(
            idx_base.index,
            idx_base[c],
            marker="o",
            linewidth=2.2,
            label=labels.get(c, c),
            color=COLORS.get(c),
        )
    ax.set_title("Index (Base=100 at first year in selection)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index value")
    ax.axhline(100, linewidth=1.2, color="#6B7280", linestyle="--")
    style_axis(ax)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Energy Mix (Annual Mean, Stacked)")

    mix = yearly.set_index("year")[
        ["total_wind_power", "total_renewables", "total_fossil"]
    ].copy()
    # avoid double-counting wind in renewables for stacked: show "renewables excl wind"
    mix["renewables_excl_wind"] = (
        mix["total_renewables"] - mix["total_wind_power"]
    ).clip(lower=0)
    mix_plot = mix[["total_wind_power", "renewables_excl_wind", "total_fossil"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(
        mix_plot.index,
        mix_plot["total_wind_power"],
        mix_plot["renewables_excl_wind"],
        mix_plot["total_fossil"],
        labels=["Wind", "Other renewables", "Fossil"],
        colors=["#14B8A6", "#34D399", "#F97316"],
        alpha=0.85,
    )
    ax.set_title("Annual Mean Generation Composition")
    ax.set_xlabel("Year")
    ax.set_ylabel("MW")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "Interpretation: this view highlights the long-run shift in the mix. "
        "Use the 'Drivers' tab to see how wind/fossil relate to CO₂ and exchanges."
    )

# Seasonality tab
with tab_season:
    st.subheader("Seasonality Explorer")

    metrics = [
        "gross_con",
        "total_wind_power",
        "total_renewables",
        "total_fossil",
        "co2_perk_wh",
        "net_exchange",
    ]

    st.markdown(
        "Explore **monthly** and **hourly** seasonality. Optionally add a second metric for comparison "
        "(dashed line, secondary axis)."
    )

    c0, c1, c2 = st.columns([2, 2, 1.2])
    with c0:
        metric_a = st.selectbox("Metric A", metrics, index=metrics.index("gross_con"))
    with c1:
        metric_b = st.selectbox("Metric B (optional)", ["None"] + metrics, index=0)
    with c2:
        show_heatmap = st.toggle("Hour×Month heatmap", value=True)

    monthly_a = df_f.groupby("month")[metric_a].mean()
    hourly_a = df_f.groupby("hour")[metric_a].mean()

    m_max, m_min = int(monthly_a.idxmax()), int(monthly_a.idxmin())
    h_max, h_min = int(hourly_a.idxmax()), int(hourly_a.idxmin())

    if metric_a == "net_exchange":
        st.info(
            f"**Net exchange summary**: strongest imports month = **{int(monthly_a.idxmax())}**, "
            f"strongest exports month = **{int(monthly_a.idxmin())}**. "
            f"Import peak hour = **{int(hourly_a.idxmax()):02d}:00**, "
            f"export peak hour = **{int(hourly_a.idxmin()):02d}:00**."
        )
        st.caption("Note: net_exchange > 0 = imports, net_exchange < 0 = exports.")
    else:
        st.info(
            f"**Seasonality summary for `{metric_a}`**: highest month = **{m_max}**, lowest month = **{m_min}**. "
            f"Peak hour = **{h_max:02d}:00**, trough hour = **{h_min:02d}:00**."
        )

    cA, cB = st.columns(2)

    with cA:
        # Mean ± std band
        g = df_f.groupby("month")[metric_a]
        m = g.mean()
        s = g.std()

        fig, ax = plt.subplots(figsize=(5.8, 3.7))
        ax.plot(
            m.index,
            m.values,
            marker="o",
            linewidth=2.2,
            color=COLORS.get(metric_a, "#3B82F6"),
            label=metric_a,
        )
        ax.fill_between(
            m.index,
            (m - s).values,
            (m + s).values,
            alpha=0.14,
            color=COLORS.get(metric_a, "#3B82F6"),
        )

        if metric_b != "None":
            m2 = df_f.groupby("month")[metric_b].mean()
            ax2 = ax.twinx()
            ax2.plot(
                m2.index,
                m2.values,
                marker="o",
                linestyle="--",
                linewidth=2.0,
                color=COLORS.get(metric_b, "#94A3B8"),
                label=metric_b,
            )
            ax2.set_ylabel(metric_b)
            ax2.grid(False)

        ax.set_title("Monthly seasonality (mean ± 1 std)")
        ax.set_xlabel("Month")
        ax.set_ylabel(metric_a)
        ax.set_xticks(range(1, 13))
        style_axis(ax)

        lines, labels = ax.get_legend_handles_labels()
        if metric_b != "None":
            l2, lab2 = ax2.get_legend_handles_labels()
            lines += l2
            labels += lab2
        ax.legend(lines, labels, frameon=False, loc="upper left")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with cB:
        g = df_f.groupby("hour")[metric_a]
        m = g.mean()
        s = g.std()

        fig, ax = plt.subplots(figsize=(5.8, 3.7))
        ax.plot(
            m.index,
            m.values,
            marker="o",
            linewidth=2.2,
            color=COLORS.get(metric_a, "#3B82F6"),
            label=metric_a,
        )
        ax.fill_between(
            m.index,
            (m - s).values,
            (m + s).values,
            alpha=0.14,
            color=COLORS.get(metric_a, "#3B82F6"),
        )

        if metric_b != "None":
            m2 = df_f.groupby("hour")[metric_b].mean()
            ax2 = ax.twinx()
            ax2.plot(
                m2.index,
                m2.values,
                marker="o",
                linestyle="--",
                linewidth=2.0,
                color=COLORS.get(metric_b, "#94A3B8"),
                label=metric_b,
            )
            ax2.set_ylabel(metric_b)
            ax2.grid(False)

        ax.set_title("Hourly seasonality (mean ± 1 std)")
        ax.set_xlabel("Hour")
        ax.set_ylabel(metric_a)
        ax.set_xticks(range(0, 24, 2))
        style_axis(ax)

        lines, labels = ax.get_legend_handles_labels()
        if metric_b != "None":
            l2, lab2 = ax2.get_legend_handles_labels()
            lines += l2
            labels += lab2
        ax.legend(lines, labels, frameon=False, loc="upper left")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    if show_heatmap:
        st.subheader("Hour × Month Pattern (mean)")
        pivot = df_f.pivot_table(
            index="hour", columns="month", values=metric_a, aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(10, 3.6))
        is_diverging = metric_a in ["net_exchange"]
        sns.heatmap(
            pivot,
            cmap="RdBu_r",
            ax=ax,
            center=0 if is_diverging else None,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"`{metric_a}`- mean by hour and month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Hour")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.caption(
        "Tip: Compare `gross_con` vs `co2_perk_wh` to see whether peak demand hours coincide with higher carbon intensity."
    )

# Drivers tab
with tab_drivers:
    st.subheader("Correlations (System Drivers)")

    corr_cols = [
        "gross_con",
        "total_wind_power",
        "total_renewables",
        "total_fossil",
        "co2_perk_wh",
        "net_exchange",
    ]
    corr = df_f[corr_cols].corr(numeric_only=True)
    order = corr["co2_perk_wh"].sort_values(ascending=False).index
    corr = corr.loc[order, order]

    co2_corr = corr["co2_perk_wh"].drop("co2_perk_wh").sort_values(ascending=False)

    top_pos = co2_corr.head(2)
    top_neg = co2_corr.tail(2)

    st.info(
        f"""
    **CO₂ drivers (correlation)**  
    Positive: {top_pos.index[0]} ({top_pos.iloc[0]:.2f}), {top_pos.index[1]} ({top_pos.iloc[1]:.2f})  
    Negative: {top_neg.index[0]} ({top_neg.iloc[0]:.2f}), {top_neg.index[1]} ({top_neg.iloc[1]:.2f})
    """
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            center=0,
            cmap="RdBu_r",
            ax=ax,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlation heatmap")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Scatter presets")
    presets = {
        "Wind vs CO₂": ("total_wind_power", "co2_perk_wh"),
        "Fossil vs CO₂": ("total_fossil", "co2_perk_wh"),
        "Wind vs Net exchange": ("total_wind_power", "net_exchange"),
        "Renewables share vs CO₂": ("renewables_share_pct", "co2_perk_wh"),
        "Renewables vs Fossil": ("total_renewables", "total_fossil"),
    }
    preset_name = st.selectbox("Preset", list(presets.keys()), index=0)
    x, y = presets[preset_name]

    c1, c2 = st.columns([3, 1])
    with c1:
        d = df_f[[x, y]].dropna()
        if len(d) > sample_n:
            d = d.sample(sample_n, random_state=42)

        q_low, q_high = 0.01, 0.99
        dx = d[x].clip(d[x].quantile(q_low), d[x].quantile(q_high))
        dy = d[y].clip(d[y].quantile(q_low), d[y].quantile(q_high))

        slope, intercept, r, p_value, std_err = linregress(dx, dy)

        fig, ax = plt.subplots(figsize=(4.5, 3))

        hb = ax.hexbin(dx, dy, gridsize=50, mincnt=1, bins="log", cmap="viridis")

        x_line = np.linspace(dx.min(), dx.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, linewidth=2.6, color="#111827")

        ax.set_title(f"{preset_name}", fontsize=10, pad=12)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=7)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=7)

        if y == "co2_perk_wh":
            ax.set_ylim(bottom=0)

        cb = fig.colorbar(hb, ax=ax, shrink=0.8)
        cb.set_label("Log density", fontsize=9)
        cb.ax.tick_params(labelsize=8)

        ax.text(
            0.80,
            0.97,
            f"r = {r:.2f}\nR² = {r**2:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            backgroundcolor="#FFFFFFAA",
            bbox=dict(boxstyle="round", alpha=0.15, edgecolor="#F4F6F7"),
        )

        style_axis(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    if preset_name == "Wind vs CO₂":
        st.caption("Higher wind generation is associated with lower carbon intensity.")
    elif preset_name == "Fossil vs CO₂":
        st.caption(
            "Fossil generation shows strong positive association with carbon intensity."
        )
    elif preset_name == "Wind vs Net exchange":
        st.caption(
            "Higher wind generation coincides with increased exports (negative net exchange)."
        )
    st.caption(
        "Reminder: correlations and regression lines here indicate association, not causal effects. "
        "System behavior is influenced by demand, weather, and cross-border flows."
    )

# Extremes tab
with tab_extremes:
    st.subheader("Outliers / Extremes (IQR rule)")

    out_cols = [
        "gross_con",
        "total_wind_power",
        "total_fossil",
        "co2_perk_wh",
        "wind_share_pct",
        "renewables_share_pct",
    ]
    out_df = iqr_outliers(df_f, out_cols)
    st.dataframe(out_df, use_container_width=True)

    st.markdown("**Boxplot explorer**")
    v = st.selectbox("Variable", out_cols, index=out_cols.index("co2_perk_wh"))

    fig, ax = plt.subplots(figsize=(7, 2.8))
    sns.boxplot(
        x=df_f[v].dropna(),
        ax=ax,
        color=COLORS.get(v, "#3B82F6"),
        width=0.45,
        fliersize=2,
    )
    ax.set_title(f"Boxplot: {v}")
    style_axis(ax, ygrid=False)
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "In electricity systems, extremes often reflect real operational events (storms, peaks, dispatch constraints). "
        "Outliers should not be removed automatically without domain justification."
    )

# Trends tab
with tab_trends:
    st.subheader("Annual Means + YoY % Change")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["total_wind_power", "total_fossil", "co2_perk_wh"]):
        sns.lineplot(
            data=yearly,
            x="year",
            y=metric,
            marker="o",
            linewidth=2.2,
            color=COLORS.get(metric),
            ax=ax,
        )
        ax.set_title(f"Yearly mean: {metric}")
        ax.set_xlabel("Year")
        style_axis(ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Year-over-year % change**")
    metrics = [
        "gross_con",
        "total_wind_power",
        "total_renewables",
        "total_fossil",
        "co2_perk_wh",
    ]
    yoy_df = yearly[["year"] + metrics].copy()
    yoy_df[metrics] = yoy_df[metrics].pct_change() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in [
        ("total_wind_power", "Wind"),
        ("total_fossil", "Fossil"),
        ("co2_perk_wh", "CO₂ intensity"),
        ("gross_con", "Consumption"),
    ]:
        sns.lineplot(
            data=yoy_df,
            x="year",
            y=col,
            marker="o",
            linewidth=2.2,
            label=label,
            color=COLORS.get(col),
            ax=ax,
        )
    ax.axhline(0, linewidth=1.2, color="#6B7280", linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("YoY % change")
    ax.set_title("YoY Percentage Change")
    style_axis(ax)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    st.pyplot(fig)
