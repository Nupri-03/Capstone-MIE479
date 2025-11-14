# -*- coding: utf-8 -*-
"""
Identify likely solar generators based on bidding behavior.
Improved version: adjusted thresholds, removed overly strict filters,
and added diagnostics to verify solar-hour coverage and candidate counts.
"""

import pandas as pd
import datetime as dt
import os
import glob

LOCAL_FOLDER_PATH = "/Users/juliarice/Desktop/capstone"

# -------------------------------------------------------
# LOAD INPUT FILES
# -------------------------------------------------------

print("Loading input files...")

asset_type = pd.read_csv(LOCAL_FOLDER_PATH + "/asset_xref_202509121109.csv")
renewables = pd.read_csv(LOCAL_FOLDER_PATH + "/caiso_sld_ren_fcst_202509191057.csv")
pricing = pd.read_csv(LOCAL_FOLDER_PATH + "/_SELECT_s_x_FROM_west_fin_ice_settlement_s_JOIN_west_fin_ice_xre_202509191043.csv")

electricity_pricing = pricing[pricing['exchangecode'].isin(["SQP", "DPN", "SDP", "UNP"])]
gas_pricing = pricing[pricing['exchangecode'].isin(["HHD", "PIG", "SCS"])]

# Load 1 year of daily bid data
folder_path = LOCAL_FOLDER_PATH + "/2023 DAM Bid Data"
files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

x = 365
files_to_load = files[:x]

print(f"Loading {len(files_to_load)} daily CSVs...")
daily_dfs = [pd.read_csv(file) for file in files_to_load]
one_day = pd.concat(daily_dfs, ignore_index=True)

# Merge with asset types
asset_day = pd.merge(asset_type, one_day, on='asset_xref', how='inner')
asset_day = asset_day[asset_day['resource_type'] == 'GENERATOR']
asset_day = asset_day.dropna(subset=['dyn_datetime_int'])

# Hour-ending setup
asset_day['dyn_datetime_int'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(asset_day['dyn_datetime_int'], unit='h')
asset_day['hour_ending'] = asset_day['dyn_datetime_int'].dt.hour + 1
asset_day = asset_day.drop(
    ['schedulingcoordinator_seq', 'mineohstateofcharge', 
     'maxeohstateofcharge', 'dyn_datetime_int', 'resourcebid_seq'],
    axis=1
)

print("✅ Finished loading input data")

# -------------------------------------------------------
# RENEWABLE (SOLAR) HOURS PROCESSING
# -------------------------------------------------------

print("\nProcessing solar hours...")

renewables = renewables[renewables['market_run_id'] == 'DAM']
hour_cols = [f'he{i:02d}' for i in range(1, 25)]

solar = renewables[renewables['renewable_type'] == 'SOLAR']
solar_sum = solar.groupby('opr_date')[hour_cols].sum().reset_index()

# Dynamic solar generation threshold
solar_threshold = 1000  # Adjust as needed (500–2000 typical)
solar_sum['solar_hours'] = solar_sum[hour_cols].apply(
    lambda row: [col for col in hour_cols if row[col] > solar_threshold], axis=1
)

# Diagnostics
avg_solar_hours = solar_sum['solar_hours'].apply(len).mean()
print(f"Average number of solar hours per day: {avg_solar_hours:.2f}")

solar_hours_by_day = solar_sum[['opr_date', 'solar_hours']]

print("✅ Finished solar hours processing")

# -------------------------------------------------------
# SOLAR CANDIDATE IDENTIFICATION
# -------------------------------------------------------

def identify_solar_candidates(df, solar_hours_by_day,
                              min_solar_output_hours=5,
                              max_offsolar_output_hours=4,
                              solar_day_fraction_threshold=0.3):
    """
    Identify solar generator candidates based on hourly bid patterns.
    """

    df = df.copy()
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['hour_ending'] = pd.to_numeric(df['hour_ending'], errors='coerce')
    df = df.dropna(subset=['quantity', 'hour_ending'])

    # Merge solar hour info
    df = df.merge(solar_hours_by_day, on='opr_date', how='left')

    def classify_day(g):
        solar_hours = g['solar_hours'].iloc[0] if isinstance(g['solar_hours'].iloc[0], list) else []
        g['he_col'] = g['hour_ending'].apply(lambda h: f'he{int(h):02d}')
        solar_mask = g['he_col'].isin(solar_hours)
        non_solar_mask = ~solar_mask

        solar_output_hours = (g.loc[solar_mask, 'quantity'] > 0).sum()
        offsolar_output_hours = (g.loc[non_solar_mask, 'quantity'] > 0).sum()

        return pd.Series({
            'solar_output_hours': solar_output_hours,
            'offsolar_output_hours': offsolar_output_hours,
            'is_solar_day': (solar_output_hours >= min_solar_output_hours) and
                            (offsolar_output_hours <= max_offsolar_output_hours)
        })

    summary = (
        df.groupby(['asset_xref', 'opr_date'])
          .apply(classify_day)
          .reset_index()
    )

    solar_candidates = (
        summary.groupby('asset_xref')['is_solar_day']
        .mean()
        .reset_index()
        .rename(columns={'is_solar_day': 'solar_day_fraction'})
    )

    solar_candidates['is_solar_candidate'] = (
        solar_candidates['solar_day_fraction'] >= solar_day_fraction_threshold
    )

    return solar_candidates, summary

print("\nIdentifying solar candidates...")
solar_candidates, solar_summary = identify_solar_candidates(asset_day, solar_hours_by_day)
print("✅ Finished solar candidate identification")

# -------------------------------------------------------
# DIAGNOSTICS / OUTPUT
# -------------------------------------------------------

positive_solar = solar_candidates[solar_candidates["is_solar_candidate"]]
non_solar = solar_candidates[~solar_candidates["is_solar_candidate"]]

print(f"\n🔆 Identified {len(positive_solar)} solar candidates out of {len(solar_candidates)} total assets.")
print("\nTop 10 solar-like assets by solar_day_fraction:")
print(positive_solar)

# Optional diagnostic histogram (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    plt.hist(solar_candidates['solar_day_fraction'], bins=20)
    plt.title("Distribution of Solar-Like Day Fractions")
    plt.xlabel("Solar Day Fraction")
    plt.ylabel("Number of Assets")
    plt.show()
except ImportError:
    pass

# -------------------------------------------------------
# OPTIONAL: SAVE RESULTS
# -------------------------------------------------------

# Uncomment if you want to save the results to CSV
# output_path = LOCAL_FOLDER_PATH + "/solar_candidates_output.csv"
# solar_candidates.to_csv(output_path, index=False)
# print(f"\nResults saved to: {output_path}")
