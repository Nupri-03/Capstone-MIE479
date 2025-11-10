#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:48:38 2025

@author: nupri
"""

import pandas as pd
import numpy as np
import os
import glob

BASE = "/Users/nupri/Desktop"
ASSET_XREF_FILE = os.path.join(BASE, "asset_xref_202509121109.csv")
RENEWABLES_FILE = os.path.join(BASE, "caiso_sld_ren_fcst_202509191057.csv")
BID_FOLDER = os.path.join(BASE, "2023 DAM Bid Data")
OUT_FILE = os.path.join(BASE, "likely_wind_candidates.csv")
LOOKBACK_DAYS = 365

def safe_read_csv(path, **kwargs):
    try: return pd.read_csv(path, **kwargs)
    except: return pd.DataFrame()

asset_type = safe_read_csv(ASSET_XREF_FILE)
renewables = safe_read_csv(RENEWABLES_FILE)

def load_bid_data(folder_path, x_days=365):
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    files = files[:min(len(files), x_days)]
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

bid_df = load_bid_data(BID_FOLDER, LOOKBACK_DAYS)

if not asset_type.empty:
    df = pd.merge(asset_type, bid_df, on="asset_xref", how="inner")
    if "resource_type" in df.columns:
        df = df[df["resource_type"].str.upper()=="GENERATOR"].copy()
else:
    df = bid_df.copy()

if "opr_date" in df.columns:
    df["opr_date"] = pd.to_datetime(df["opr_date"], errors="coerce")
elif "dyn_datetime_int" in df.columns:
    df["dyn_datetime_int"] = pd.to_numeric(df["dyn_datetime_int"], errors="coerce")
    df["opr_date"] = (pd.to_datetime("1900-01-01") +
                      pd.to_timedelta(df["dyn_datetime_int"], unit="h")).dt.floor("D")

df["hour_ending"] = pd.to_numeric(df.get("hour_ending", np.nan), errors="coerce")
df["quantity"] = pd.to_numeric(df.get("quantity", 0), errors="coerce").fillna(0)
df["selfschedmw"] = pd.to_numeric(df.get("selfschedmw", 0), errors="coerce").fillna(0)
df["gen_mw"] = df["quantity"] + df["selfschedmw"]

df["month"] = df["opr_date"].dt.month
season_map = {3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"summer",
              10:"fall",11:"fall",12:"winter",1:"winter",2:"winter"}
df["season"] = df["month"].map(season_map)

def compute_seasonal_stats(df):
    grouped = df.groupby(["asset_xref", "season"])["gen_mw"].mean().unstack(fill_value=0)
    grouped["mean_all"] = grouped.mean(axis=1).replace(0, np.nan)
    eps = 1e-6

    # Ratios relative to annual mean
    grouped["summer_mean_ratio"] = grouped.get("summer", 0) / (grouped["mean_all"] + eps)
    grouped["winter_mean_ratio"] = grouped.get("winter", 0) / (grouped["mean_all"] + eps)
    grouped["summer_vs_spring"] = grouped.get("summer", 0) / (grouped.get("spring", 0) + eps)

    # Expected: wind farms show lower winter CF (~15%) vs annual (~26%)
    # i.e., winter_mean_ratio ≈ 0.6 of annual mean (15/26 ≈ 0.58)
    grouped["winter_deficit_ratio"] = grouped["winter_mean_ratio"] / (grouped["summer_mean_ratio"] + eps)
    
    return grouped.reset_index()
def compute_selfsched_fraction(df):
    df2 = df.copy()
    df2["is_selfsched"] = df2["selfschedmw"] > 0
    agg = df2.groupby("asset_xref").agg(
        total_hours=("is_selfsched","size"),
        selfsched_hours=("is_selfsched","sum"),
        mean_gen=("gen_mw","mean"),
        std_gen=("gen_mw","std")
    ).reset_index()
    agg["selfsched_frac"] = agg["selfsched_hours"]/agg["total_hours"]
    agg["volatility"] = agg["std_gen"]/(agg["mean_gen"].abs()+1e-6)
    return agg

seasonal = compute_seasonal_stats(df)
selfsched = compute_selfsched_fraction(df)
candidates = seasonal.merge(selfsched, on="asset_xref", how="left")


# ---------- WIND DECISION RULE ----------

# Wind decision rule (tunable). combination of seasonal pattern, selfsched fraction,
# volatility, mean generation 
# The average generation in summer months (June–Sept) is at least 25% higher than the asset’s average across the whole year.
# The asset was self-scheduled at least 30% of the hours it was operating.
# The average generation (in MW) must be at least 0.5 MW.
# Wind tends to be volatile; thermal or hydro tends to be more stable, therefore higher volatility


candidates["is_likely_wind"] = (
    (candidates["summer_mean_ratio"] >= 1.25) &              # summer output ≥ 125% of annual
    (candidates["winter_mean_ratio"] <= 0.75) &              # winter ≤ 75% of annual mean
    (candidates["winter_deficit_ratio"] <= 0.7) &            # winter ≤ 70% of summer
    (candidates["selfsched_frac"] >= 0.30) &                 # ≥ 30% hours self-scheduled
    (candidates["mean_gen"] >= 0.5) &                        # average ≥ 0.5 MW
    (candidates["volatility"] >= 0.15)                       # ≥ 15% relative std dev
)


likely_wind = candidates[candidates["is_likely_wind"]].copy()
likely_wind.to_csv(OUT_FILE, index=False)

print("Likely wind assets:", len(likely_wind))
print(likely_wind[["asset_xref","summer_mean_ratio","selfsched_frac"]].head())
