#!/usr/bin/env python3
"""Exploratory analysis for Occuspace 30-min exports (Cal Poly Rec Center)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent
OUT = BASE / "eda_output"
OUT.mkdir(exist_ok=True)

FILES = [
    ("2023-05_to_2024-05", "-30minExport-1May23-1May24.csv"),
    ("2024-06_to_2025-06", "-30minExport-1Jun24-1Jun25.csv"),
    ("2025-06_to_2026-04", "-30minExport-1Jun25-15Apr26.csv"),
]


def load_export(period: str, name: str) -> pd.DataFrame:
    path = BASE / name
    # First 6 lines: metadata + blank; line 7 (0-indexed 6) is the column header
    df = pd.read_csv(path, skiprows=6)
    df["export_period"] = period
    df["source_file"] = name
    return df


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120

    frames = [load_export(p, n) for p, n in FILES]
    df = pd.concat(frames, ignore_index=True)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %H:%M")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    numeric = [
        "Day of Week",
        "Week of Year",
        "Hour of Day",
        "Average Occupancy",
        "Average Utilization",
        "Peak Occupancy",
        "Peak Utilization",
        "Capacity",
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Report
    lines = []
    lines.append("# Occuspace EDA summary\n")
    lines.append(f"Total rows (all exports): {len(df):,}\n")
    lines.append(f"Columns: {list(df.columns)}\n")
    lines.append("\n## Rows per export period\n")
    lines.append(df["export_period"].value_counts().sort_index().to_string())
    lines.append("\n\n## Locations\n")
    lines.append(
        df.groupby("Location")
        .agg(
            rows=("Location", "count"),
            capacity=("Capacity", "first"),
            date_min=("Date", "min"),
            date_max=("Date", "max"),
        )
        .to_string()
    )
    lines.append("\n\n## Numeric describe (all data)\n")
    lines.append(df[numeric].describe().round(4).to_string())
    lines.append("\n\n## Missing values (selected)\n")
    lines.append(df[numeric + ["Location", "Timestamp"]].isna().sum().to_string())

    lines.append("\n\n## Data quality notes\n")
    lines.append(
        f"Rows with Average Utilization > 1: {(df['Average Utilization'] > 1.0).sum():,} "
        f"({100 * (df['Average Utilization'] > 1.0).mean():.2f}%)\n"
    )
    lines.append(
        f"Rows with Peak Utilization > 1: {(df['Peak Utilization'] > 1.0).sum():,} "
        f"({100 * (df['Peak Utilization'] > 1.0).mean():.2f}%)\n"
    )
    lines.append(
        "Interpretation: values above 100% utilization suggest capacity metadata may not "
        "match peak sensor counts, or definitions differ between Occuspace fields.\n"
    )

    summary_path = OUT / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # --- Figures ---
    locs = sorted(df["Location"].dropna().unique())
    palette = sns.color_palette("husl", n_colors=max(len(locs), 3))

    # 1) Utilization by hour (overall)
    fig, ax = plt.subplots(figsize=(9, 4))
    hourly = (
        df.groupby(["Hour of Day", "Location"], observed=True)["Average Utilization"]
        .median()
        .reset_index()
    )
    sns.lineplot(
        data=hourly,
        x="Hour of Day",
        y="Average Utilization",
        hue="Location",
        ax=ax,
        marker="o",
    )
    ax.set_title("Median Average Utilization by Hour of Day")
    ax.set_ylabel("Median utilization")
    fig.tight_layout()
    fig.savefig(OUT / "01_utilization_by_hour.png")
    plt.close(fig)

    # 2) By day of week
    fig, ax = plt.subplots(figsize=(9, 4))
    dow = (
        df.groupby(["Day of Week", "Location"], observed=True)["Average Utilization"]
        .median()
        .reset_index()
    )
    sns.barplot(
        data=dow,
        x="Day of Week",
        y="Average Utilization",
        hue="Location",
        ax=ax,
    )
    ax.set_title("Median Average Utilization by Day of Week (1=Sun … 7=Sat)")
    fig.tight_layout()
    fig.savefig(OUT / "02_utilization_by_dow.png")
    plt.close(fig)

    # 3) Distribution of Average Utilization
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, loc in enumerate(locs):
        sub = df.loc[df["Location"] == loc, "Average Utilization"].dropna()
        sns.kdeplot(sub, ax=ax, label=loc, color=palette[i % len(palette)])
    ax.set_title("Distribution of Average Utilization (KDE by location)")
    ax.set_xlabel("Average Utilization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "03_kde_utilization.png")
    plt.close(fig)

    # 4) Time series sample (recent month slice per period — last 45 days of max date)
    fig, axes = plt.subplots(len(locs), 1, figsize=(11, 3 * len(locs)), sharex=True)
    if len(locs) == 1:
        axes = [axes]
    tmax = df["Timestamp"].max()
    tmin = tmax - pd.Timedelta(days=45)
    ts = df[(df["Timestamp"] >= tmin) & (df["Timestamp"] <= tmax)]
    for ax, loc in zip(axes, locs):
        sub = ts[ts["Location"] == loc].sort_values("Timestamp")
        ax.plot(sub["Timestamp"], sub["Average Occupancy"], lw=0.6, alpha=0.85)
        ax.set_ylabel(loc[:30])
    axes[0].set_title(f"Average Occupancy (last ~45 days ending {tmax.date()})")
    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(OUT / "04_timeseries_recent.png")
    plt.close(fig)

    # 5) Heatmap: hour x day of week (median util), one panel per location
    n = len(locs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, loc in zip(axes[0], locs):
        sub = df[df["Location"] == loc]
        pivot = sub.pivot_table(
            values="Average Utilization",
            index="Hour of Day",
            columns="Day of Week",
            aggfunc="median",
        )
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Median util"})
        ax.set_title(loc)
        ax.set_xlabel("Day of week (1=Sun)")
    fig.suptitle("Median Average Utilization: Hour × Day of Week", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "05_heatmap_hour_dow.png", bbox_inches="tight")
    plt.close(fig)

    # 6) Rows per calendar month (volume / coverage)
    df["year_month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby(["year_month", "Location"]).size().reset_index(name="rows")
    monthly["year_month"] = monthly["year_month"].astype(str)
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.lineplot(data=monthly, x="year_month", y="rows", hue="Location", marker="o", ax=ax)
    ax.set_title("Observation count per month by location")
    ax.set_ylabel("Row count")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "06_monthly_row_counts.png")
    plt.close(fig)

    print(f"Wrote summaries and figures to {OUT}")


if __name__ == "__main__":
    main()
