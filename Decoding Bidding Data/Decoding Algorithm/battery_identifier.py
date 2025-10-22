# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 06:56:57 2025

@author: jliu
"""

import pandas as pd
import datetime as dt
import os
import glob

asset_type = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\asset_xref_202509121109.csv")

renewables = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\caiso_sld_ren_fcst_202509191057.csv")

pricing = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\_SELECT_s_x_FROM_west_fin_ice_settlement_s_JOIN_west_fin_ice_xre_202509191043.csv")
electricty_pricing = pricing[pricing['exchangecode'].isin(["SQP", "DPN", "SDP", "UNP"])]
gas_pricing = pricing[pricing['exchangecode'].isin(["HHD", "PIG", "SCS"])]

#Load in x number of days worth of data
folder_path = r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\2023 DAM Bid Data"
files = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) 

x = 365 
files_to_load = files[:x]  

daily_dfs = [pd.read_csv(file) for file in files_to_load]
one_day = pd.concat(daily_dfs, ignore_index=True)

asset_day = pd.merge(asset_type, one_day, on='asset_xref', how='inner')
asset_day = asset_day[asset_day['resource_type'] == 'GENERATOR']
asset_day = asset_day.dropna(subset=['dyn_datetime_int'])

#Sort out the Hour-Ending Aspect Based on Chat Feedback
asset_day['dyn_datetime_int'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(asset_day['dyn_datetime_int'], unit='h')
asset_day['hour_ending'] = asset_day['dyn_datetime_int'].dt.hour
asset_day['hour_ending'] = asset_day['hour_ending'] + 1
asset_day = asset_day.drop(['schedulingcoordinator_seq', 'mineohstateofcharge', 'maxeohstateofcharge', 'dyn_datetime_int', 'resourcebid_seq'], axis=1)

#RENEWABLES Hour Ending 
renewables = renewables[renewables['market_run_id'] == 'DAM']

he_cols = [col for col in renewables.columns if col.startswith('he')]
renew_long = renewables.melt(
    id_vars=['opr_date', 'trading_hub', 'renewable_type'],
    value_vars=he_cols,
    var_name='hour_ending',
    value_name='mw'
)

renew_long['hour_ending'] = renew_long['hour_ending'].str.extract(r'(\d+)').astype(int)

renew_long['hub_type'] = renew_long['trading_hub'] + "_" + renew_long['renewable_type']

renew_pivot = renew_long.pivot_table(
    index=['opr_date', 'hour_ending'],
    columns='hub_type',
    values='mw',
    aggfunc='sum'
).reset_index()

print(renew_pivot.head())

solar = renewables[renewables['renewable_type'] == 'SOLAR']
hour_cols = [f'he{i:02d}' for i in range(1, 25)]  # creates ['he01', ..., 'he24']
solar_sum = solar.groupby('opr_date')[hour_cols].sum().reset_index()

# List of hour columns
hour_cols = [f'he{i:02d}' for i in range(1, 25)]

# For each day, list which hours exceed 5000
solar_sum['solar_hours'] = solar_sum[hour_cols].apply(
    lambda row: [col for col in hour_cols if row[col] > 5000], axis=1
)

# Display opr_date and identified solar hours
solar_hours_by_day = solar_sum[['opr_date', 'solar_hours']]

def identify_battery_candidates(df, solar_hours_by_day):
    """
    Identify battery candidates based on daily solar-hour charging/discharging behavior.

    Parameters
    ----------
    df : DataFrame
        Daily asset-level bid data, must include columns:
        ['asset_xref', 'opr_date', 'hour_ending', 'quantity']
    solar_hours_by_day : DataFrame
        Contains columns ['opr_date', 'solar_hours'] with solar hours per day

    Returns
    -------
    battery_candidates : DataFrame
        Asset-level classification of likely batteries
    summary : DataFrame
        Daily-level classification details
    """

    df = df.copy()
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['hour_ending'] = pd.to_numeric(df['hour_ending'], errors='coerce')
    df = df.dropna(subset=['quantity', 'hour_ending'])

    # Merge solar hour info to main df
    df = df.merge(solar_hours_by_day, on='opr_date', how='left')

    # Keep only days where at least 6 hours have significant solar generation
    df['solar_hour_count'] = df['solar_hours'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df = df[df['solar_hour_count'] >= 6]

    def classify_day(g):
        # Get the list of solar hours for this date
        solar_hours = g['solar_hours'].iloc[0] if isinstance(g['solar_hours'].iloc[0], list) else []
        if len(solar_hours) < 6:
            return pd.Series({
                'solar_hours_count': len(solar_hours),
                'charge_hours': 0,
                'discharge_hours': 0,
                'is_battery_day': False
            })

        # Convert numeric hour_ending → 'he01' format to compare with solar_hours list
        g['he_col'] = g['hour_ending'].apply(lambda h: f'he{int(h):02d}')

        # Define masks
        solar_mask = g['he_col'].isin(solar_hours)
        non_solar_mask = ~solar_mask

        # Count charge/discharge behavior
        charge_hours = (g.loc[solar_mask, 'quantity'] < 0).sum()      # charging (negative quantity)
        discharge_hours = (g.loc[non_solar_mask, 'quantity'] > 0).sum()  # discharging (positive quantity)

        return pd.Series({
            'solar_hours_count': len(solar_hours),
            'charge_hours': charge_hours,
            'discharge_hours': discharge_hours,
            'is_battery_day': (charge_hours >= 3) and (discharge_hours >= 3)
        })

    # Apply per asset per day
    summary = (
        df.groupby(['asset_xref', 'opr_date'])
          .apply(classify_day)
          .reset_index()
    )

    # Aggregate to asset level — if the unit behaves like a battery on any day
    battery_candidates = (
        summary.groupby('asset_xref')['is_battery_day']
        .any()
        .reset_index()
        .rename(columns={'is_battery_day': 'is_battery_candidate'})
    )

    return battery_candidates, summary


battery_candidates_1, battery_summary = identify_battery_candidates(asset_day, solar_hours_by_day)

def summarize_bids(df):
    # Identify bid and quantity columns
    bid_cols = [col for col in df.columns if col.lower().startswith('price')]
    qty_cols = [col for col in df.columns if col.lower().startswith('quantity')]

    # Group by asset/date/hour and take the mean (or sum if appropriate)
    summary = (
        df.groupby(['asset_xref', 'opr_date', 'hour_ending'])[bid_cols + qty_cols]
        .mean()
        .reset_index()
    )

    # Reshape for visualization
    bid_long = summary.melt(
        id_vars=['asset_xref', 'opr_date', 'hour_ending'],
        var_name='tier',
        value_name='value'
    )

    # Extract tier type (price/quantity)
    bid_long['tier_type'] = bid_long['tier'].str.extract(r'([a-zA-Z]+)')

    # Extract tier level safely (some may not have a number)
    bid_long['tier_level'] = (
        bid_long['tier'].str.extract(r'(\d+)')
        .astype(float)  # convert safely to float first
    )

    return summary, bid_long

sql_code = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\_select_from_select_from_asset_xref_ax_resource_xref_rx_select_d_202510211016.csv")

# Filter battery_candidates_1 for rows where is_battery_candidate is False
non_battery = battery_candidates_1[battery_candidates_1['is_battery_candidate'] == False]

# Find rows in sql_code whose asset_xref is in non_battery
result = sql_code[sql_code['asset_xref'].isin(non_battery['asset_xref'])]