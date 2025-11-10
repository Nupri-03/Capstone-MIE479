#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 19:57:00 2025

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

# correlation thresholds / parameters
MIN_MATCHED_POINTS_FOR_CORR = 30   
MIN_STD_FOR_CORR = 1e-6           

def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return pd.DataFrame()

def load_bid_data(folder_path, x_days=365):
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    files = files[:min(len(files), x_days)]
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: couldn't read {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# -------------------------------------------------------------------
# Load files
asset_type = safe_read_csv(ASSET_XREF_FILE)
renewables = safe_read_csv(RENEWABLES_FILE)
bid_df = load_bid_data(BID_FOLDER, LOOKBACK_DAYS)

# -------------------------------------------------------------------
# Merge xref if available, keep only GENERATOR resources if resource_type exists
if not asset_type.empty and "asset_xref" in asset_type.columns:
    df = pd.merge(asset_type, bid_df, on="asset_xref", how="inner")
    if "resource_type" in df.columns:
        # uppercase comparison; keep rows labeled GENERATOR (defensive)
        df = df[df["resource_type"].astype(str).str.upper() == "GENERATOR"].copy()
else:
    df = bid_df.copy()

# -------------------------------------------------------------------
# Normalize/derive datetime and hour_ending on df (bid data)

if "opr_date" in df.columns:
    df["opr_date"] = pd.to_datetime(df["opr_date"], errors="coerce")
elif "dyn_datetime_int" in df.columns:
    # dyn_datetime_int appears to be hours since 1900-01-01
    df["dyn_datetime_int"] = pd.to_numeric(df["dyn_datetime_int"], errors="coerce")
    df["opr_date"] = (pd.to_datetime("1900-01-01") +
                      pd.to_timedelta(df["dyn_datetime_int"], unit="h")).dt.floor("D")
else:
    # if no datetime, create an empty column and warn
    df["opr_date"] = pd.NaT
    print("Warning: no opr_date or dyn_datetime_int in bid data.")

# hour_ending: try multiple possible columns
if "hour_ending" in df.columns:
    df["hour_ending"] = pd.to_numeric(df["hour_ending"], errors="coerce")
else:
    # try to derive from dyn_datetime_int if present
    if "dyn_datetime_int" in df.columns:
        try:
            dt_col = pd.to_datetime(df["dyn_datetime_int"], unit='h', origin=pd.Timestamp("1900-01-01"))
            df["hour_ending"] = dt_col.dt.hour + 1  # convert to hour ending 1..24
        except Exception:
            df["hour_ending"] = np.nan
    else:
        df["hour_ending"] = np.nan

# Clean hour_ending to be integers 1..24 (coerce and map 0->24 if needed)

df["hour_ending"] = df["hour_ending"].round(0).astype('Int64')
# Some sources use 0 for midnight; convert 0->24
df.loc[df["hour_ending"] == 0, "hour_ending"] = 24

# normalize quantity/selfsched and compute gen_mw
df["quantity"] = pd.to_numeric(df.get("quantity", 0), errors="coerce").fillna(0)
df["selfschedmw"] = pd.to_numeric(df.get("selfschedmw", 0), errors="coerce").fillna(0)
df["gen_mw"] = df["quantity"] + df["selfschedmw"]

# add month/season
df["month"] = df["opr_date"].dt.month
season_map = {3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"summer",
              10:"fall",11:"fall",12:"winter",1:"winter",2:"winter"}
df["season"] = df["month"].map(season_map)

# -------------------------------------------------------------------
# Seasonal statistics 

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

# Self-schedule fraction / volatility
def compute_selfsched_fraction(df):
    df2 = df.copy()
    df2["is_selfsched"] = df2["selfschedmw"] > 0
    agg = df2.groupby("asset_xref").agg(
        total_hours=("is_selfsched","size"),
        selfsched_hours=("is_selfsched","sum"),
        mean_gen=("gen_mw","mean"),
        std_gen=("gen_mw","std")
    ).reset_index()
    agg["selfsched_frac"] = agg["selfsched_hours"]/agg["total_hours"].replace(0,np.nan)
    agg["volatility"] = agg["std_gen"]/(agg["mean_gen"].abs()+1e-6)
    return agg

seasonal = compute_seasonal_stats(df)
selfsched = compute_selfsched_fraction(df)
candidates = seasonal.merge(selfsched, on="asset_xref", how="left")

# -------------------------------------------------------------------
# Wind correlation feature:
    
wind_corr = pd.DataFrame(columns=["asset_xref","wind_corr","n_pairs"])

if not renewables.empty:
    # keep only DAM run forecasts if column exists
    if "market_run_id" in renewables.columns:
        renew_filtered = renewables[renewables["market_run_id"] == "DAM"].copy()
    else:
        renew_filtered = renewables.copy()

    # Identify hour columns 
    he_cols = [c for c in renew_filtered.columns if str(c).lower().startswith("he")]
    if he_cols:
        # melt to long format
        id_vars = [c for c in ["opr_date", "trading_hub", "renewable_type"] if c in renew_filtered.columns]
        renew_long = renew_filtered.melt(id_vars=id_vars, value_vars=he_cols,
                                         var_name="hour_ending_raw", value_name="mw")
        # Extract numeric hour from 'he01' etc
        renew_long["hour_ending"] = renew_long["hour_ending_raw"].astype(str).str.extract(r"(\d+)").astype(float).round(0).astype('Int64')
        renew_long.loc[renew_long["hour_ending"] == 0, "hour_ending"] = 24

        # restrict to wind rows
        if "renewable_type" in renew_long.columns:
            wind_long = renew_long[renew_long["renewable_type"].astype(str).str.upper() == "WIND"].copy()
        else:
            # fallback: any hub names with 'wind' in them
            cols_joined = renew_long.columns.astype(str).str.lower()
            wind_long = renew_long[renew_long.astype(str).apply(lambda row: row.str.contains("wind").any(), axis=1)]

        # Ensure opr_date is datetime
        if "opr_date" in wind_long.columns:
            wind_long["opr_date"] = pd.to_datetime(wind_long["opr_date"], errors="coerce")
        else:
            wind_long["opr_date"] = pd.NaT

        # create hub-level name (hub+type) if available to average across hubs
        if "trading_hub" in wind_long.columns and "renewable_type" in wind_long.columns:
            wind_long["hub_type"] = wind_long["trading_hub"].astype(str) + "_" + wind_long["renewable_type"].astype(str)
        else:
            wind_long["hub_type"] = wind_long.get("trading_hub", "hub").astype(str)

        # pivot so we get a single wind_fcst per opr_date/hour_ending by averaging across hubs
        wind_pivot = (wind_long.groupby(["opr_date", "hour_ending"])["mw"]
                                 .mean()
                                 .reset_index()
                                 .rename(columns={"mw": "wind_fcst"}))

        # Merge bid df with wind forecast (inner join to get matched pairs)
        merged = pd.merge(
            df[["asset_xref", "opr_date", "hour_ending", "gen_mw"]],
            wind_pivot,
            on=["opr_date", "hour_ending"],
            how="inner"
        )

        if not merged.empty:
            # compute correlation per asset_xref with robust checks
            def safe_corr(group):
                gen = group["gen_mw"].astype(float).to_numpy()
                wfc = group["wind_fcst"].astype(float).to_numpy()
                # drop pairs with nan
                mask = ~np.isnan(gen) & ~np.isnan(wfc)
                gen = gen[mask]
                wfc = wfc[mask]
                n = len(gen)
                if n < MIN_MATCHED_POINTS_FOR_CORR:
                    return pd.Series({"wind_corr": 0.0, "n_pairs": n})
                if np.nanstd(gen) < MIN_STD_FOR_CORR or np.nanstd(wfc) < MIN_STD_FOR_CORR:
                    # constant series -> correlation not meaningful
                    return pd.Series({"wind_corr": 0.0, "n_pairs": n})
                try:
                    corr = float(np.corrcoef(gen, wfc)[0,1])
                    if np.isnan(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0
                return pd.Series({"wind_corr": corr, "n_pairs": n})

            corr_df = merged.groupby("asset_xref").apply(safe_corr).reset_index()
            wind_corr = corr_df
        else:
            print("No matched rows between bid data and wind forecasts after merging.")
    else:
        print("No hour (heXX) columns found in renewables file; can't compute wind correlation.")
else:
    print("Renewables file empty; skipping wind correlation.")

# merge into candidates (left join)
candidates = candidates.merge(wind_corr, on="asset_xref", how="left")

# fill missing numeric columns with safe defaults
candidates["wind_corr"] = pd.to_numeric(candidates.get("wind_corr", 0), errors="coerce").fillna(0)
candidates["n_pairs"] = pd.to_numeric(candidates.get("n_pairs", 0), errors="coerce").fillna(0)

# -------------------------------------------------------------------
# Wind decision rule (tunable). combination of seasonal pattern, selfsched fraction,
# volatility, mean generation and/or wind forecast correlation.
# The average generation in summer months (June–Sept) is at least 25% higher than the asset’s average across the whole year.
# The asset was self-scheduled at least 30% of the hours it was operating.
# The average generation (in MW) must be at least 0.5 MW.
# Wind tends to be volatile; thermal or hydro tends to be more stable, therefore higher volatility

candidates["is_likely_wind"] = (
    (candidates["summer_mean_ratio"] >= 1.25) &              # summer output ≥ 125% of annual
    # (candidates["winter_mean_ratio"] <= 0.75) &            # winter ≤ 75% of annual mean
    (candidates["winter_deficit_ratio"] <= 0.8) &            # winter ≤ 70% of summer
    #(candidates["selfsched_frac"] >= 0.30) &                # ≥ 30% hours self-scheduled
    (candidates["mean_gen"] >= 0.5) &                        # average ≥ 0.5 MW
    #(candidates["volatility"] >= 0.15) &                    # ≥ 15% relative std dev
    (candidates["wind_corr"] >= 0.30)                        # should have higher correlation with forecasts
)

#  strong correlation to wind forecast (require at least MIN_MATCHED points)
#strong_corr_mask = (candidates["wind_corr"] >= 0.35) & (candidates["n_pairs"] >= MIN_MATCHED_POINTS_FOR_CORR)
#candidates.loc[strong_corr_mask, "is_likely_wind"] = True

# filter out obvious non-wind (very low volatility & near-zero selfsched fraction)
candidates.loc[(candidates["volatility"] < 0.01) & (candidates["selfsched_frac"] < 0.01), "is_likely_wind"] = False

# export results
likely_wind = candidates[candidates["is_likely_wind"] == True].copy()
likely_wind.to_csv(OUT_FILE, index=False)

# diagnostics
print("Total assets evaluated:", len(candidates))
print("Likely wind assets:", len(likely_wind))
print("Top candidates sample:")
print(likely_wind[["asset_xref", "wind_corr", "n_pairs", "summer_mean_ratio", "selfsched_frac", "mean_gen", "volatility"]].head(20))

