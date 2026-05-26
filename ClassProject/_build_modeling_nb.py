#!/usr/bin/env python3
"""Build and validate modeling pipeline; writes modeling.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
PROCESSED = BASE / "processed"
SEED = 42
GAP_THRESHOLD = pd.Timedelta(hours=8)
SLOTS_PER_DAY = 36
CAPACITY = 220

np.random.seed(SEED)


def assign_term_for_date(dates: pd.Series, terms: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=dates.index)
    out["academic_year"] = pd.NA
    out["term"] = pd.NA
    out["term_phase"] = pd.NA
    for _, row in terms.iterrows():
        instruction = dates.between(row.classes_begin, row.last_day_classes)
        finals = dates.between(row.finals_begin, row.finals_end)
        in_term = instruction | finals
        out.loc[in_term, "academic_year"] = row.academic_year
        out.loc[in_term, "term"] = row.term
        out.loc[instruction, "term_phase"] = "instruction"
        out.loc[finals, "term_phase"] = "finals"
    return out


def operating_range(day: pd.Timestamp) -> pd.DatetimeIndex:
    start = day.normalize() + pd.Timedelta(hours=6)
    end = day.normalize() + pd.Timedelta(hours=23, minutes=30)
    return pd.date_range(start, end, freq="30min")


def impute_export_gaps(df: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["is_imputed"] = False
    df["imputation_method"] = "none"

    observed = df[~df["is_imputed"]].copy()
    med_term = (
        observed.groupby(["academic_year", "term", "Day of Week", "Hour of Day"])[
            "Average Occupancy"
        ]
        .median()
    )
    train_mask = observed["academic_year"].isin(["2022-23", "2023-24", "2024-25"])
    med_global = (
        observed.loc[train_mask]
        .groupby(["Day of Week", "Hour of Day"])["Average Occupancy"]
        .median()
    )

    ts_sorted = df["Timestamp"].sort_values().unique()
    new_rows = []

    for i in range(len(ts_sorted) - 1):
        t0, t1 = ts_sorted[i], ts_sorted[i + 1]
        delta = t1 - t0
        if delta <= GAP_THRESHOLD:
            continue

        existing = set(pd.to_datetime(df["Timestamp"]))
        day = t0.normalize()
        end_day = t1.normalize()
        while day <= end_day:
            for slot in operating_range(day):
                if slot <= t0 or slot >= t1:
                    continue
                if slot in existing:
                    continue
                date_only = slot.normalize()
                term_info = assign_term_for_date(
                    pd.Series([date_only]), terms
                ).iloc[0]
                if pd.isna(term_info["academic_year"]):
                    continue
                dow = (slot.dayofweek + 1) % 7 + 1  # Occuspace: 1=Sun .. 7=Sat
                hour = slot.hour
                key_term = (
                    term_info["academic_year"],
                    term_info["term"],
                    dow,
                    hour,
                )
                if key_term in med_term.index:
                    occ = med_term.loc[key_term]
                    method = "term_dow_hour"
                else:
                    occ = med_global.loc[(dow, hour)]
                    method = "global_dow_hour"

                row_before = df[df["Timestamp"] <= t0].iloc[-1]
                new_rows.append(
                    {
                        "Location": row_before["Location"],
                        "Timestamp": slot,
                        "Date": date_only,
                        "Day of Week": dow,
                        "Hour of Day": hour,
                        "Average Occupancy": float(occ),
                        "Average Utilization": float(occ) / CAPACITY,
                        "Capacity": CAPACITY,
                        "academic_year": term_info["academic_year"],
                        "term": term_info["term"],
                        "term_phase": term_info["term_phase"],
                        "is_imputed": True,
                        "imputation_method": method,
                    }
                )
                existing.add(slot)
            day += pd.Timedelta(days=1)

    if new_rows:
        imputed = pd.DataFrame(new_rows)
        df = pd.concat([df, imputed], ignore_index=True)
        df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Timestamp").reset_index(drop=True)
    y = df["Average Occupancy"]
    df["lag_1"] = y.shift(1)
    df["lag_2"] = y.shift(2)
    df["lag_48"] = y.shift(48)
    df["lag_336"] = y.shift(336)
    df["roll_mean_4"] = y.shift(1).rolling(4, min_periods=1).mean()
    df["roll_mean_48"] = y.shift(1).rolling(48, min_periods=1).mean()
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour of Day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour of Day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["Day of Week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["Day of Week"] / 7)
    return df


def main():
    raw = pd.read_csv(PROCESSED / "occuspace_30min_rec_center.csv")
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    terms = pd.read_csv(PROCESSED / "academic_terms.csv")
    for c in ["classes_begin", "last_day_classes", "finals_begin", "finals_end"]:
        terms[c] = pd.to_datetime(terms[c])

    df = impute_export_gaps(raw, terms)
    print("After impute:", len(df), "imputed:", df["is_imputed"].sum())
    df.to_csv(PROCESSED / "occuspace_30min_rec_center_imputed.csv", index=False)

    df = add_features(df)
    train = df[df["academic_year"].isin(["2022-23", "2023-24", "2024-25"])]
    test = df[df["academic_year"] == "2025-26"]
    print("train", len(train), "test", len(test))


if __name__ == "__main__":
    main()
