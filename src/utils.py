import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "time_utc" not in df.columns:
        raise ValueError("Missing column: time_utc")

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.set_index("time_utc").sort_index()

    return df


def yoy(series: pd.Series) -> float:
    """YoY % change of last value vs previous value in an annual series."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    return (s.iloc[-1] / s.iloc[-2] - 1) * 100


def base100(df_year: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Index columns to 100 at first year."""
    out = df_year[cols].copy()
    for c in cols:
        first = out[c].dropna().iloc[0] if out[c].notna().any() else np.nan
        out[c] = (out[c] / first) * 100 if pd.notna(first) and first != 0 else np.nan
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    renewable_cols = [
        "offshore_wind_power",
        "onshore_wind_power",
        "hydro_power",
        "solar_power",
        "biomass",
        "biogas",
        "waste",
    ]
    fossil_cols = ["fossil_gas", "fossil_oil", "fossil_hard_coal"]
    exchange_cols = [
        "exchange_great_belt",
        "exchange_germany",
        "exchange_sweden",
        "exchange_norway",
        "exchange_netherlands",
        "exchange_great_britain",
    ]

    required = (
        renewable_cols + fossil_cols + exchange_cols + ["gross_con", "co2_perk_wh"]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()

    df["total_wind_power"] = df["offshore_wind_power"] + df["onshore_wind_power"]
    df["total_renewables"] = df[renewable_cols].sum(axis=1)
    df["total_fossil"] = df[fossil_cols].sum(axis=1)
    df["net_exchange"] = df[exchange_cols].sum(axis=1, min_count=1)

    # Shares (grid-consistent; NOT including self-consumption)
    df["renewables_share_pct"] = (
        100 * df["total_renewables"] / df["gross_con"]
    ).replace([np.inf, -np.inf], np.nan)
    df["wind_share_pct"] = (100 * df["total_wind_power"] / df["gross_con"]).replace(
        [np.inf, -np.inf], np.nan
    )
    df["fossil_share_pct"] = (100 * df["total_fossil"] / df["gross_con"]).replace(
        [np.inf, -np.inf], np.nan
    )

    df["year"] = df.index.year
    df["month"] = df.index.month
    df["hour"] = df.index.hour

    return df


def iqr_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out = ((s < lower) | (s > upper)).sum()
        rows.append([col, lower, upper, out, 100 * out / len(s)])
    return pd.DataFrame(
        rows, columns=["variable", "lower", "upper", "outlier_count", "outlier_pct"]
    )


def style_axis(ax: plt.Axes, ygrid: bool = True) -> None:
    ax.grid(axis="y" if ygrid else "both", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
