# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:49:50 2025

@author: jliu
"""

import pandas as pd
import datetime as dt
import os
import glob

BASE = "Data Required for CAPSTONE Modelling"

ASSET_XREF_FILE     = os.path.join(BASE, "asset_xrefs.csv")
PRICING_FILE        = os.path.join(BASE, "Energy_Gas_Prices_2024.csv")
RESOURCE_XREF_FILE  = os.path.join(BASE, "resource_xrefs.csv")
BID_FOLDER          = os.path.join(BASE, "2024 DAM Bid Data.zip")   # unzip folder here

asset_type = pd.read_csv(ASSET_XREF_FILE)

pricing = pd.read_csv(PRICING_FILE)
electricty_pricing = pricing[pricing['exchangecode'].isin(["SQP", "DPN", "SDP", "UNP"])]
gas_pricing = pricing[pricing['exchangecode'].isin(["HHD", "PIG", "SCS"])]

# Load X days of bidding data
folder_path = BID_FOLDER
files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

#asset_type = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\asset_xrefs.csv")

#pricing = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\Energy_Gas_Prices_2024.csv")
#electricty_pricing = pricing[pricing['exchangecode'].isin(["SQP", "DPN", "SDP", "UNP"])]
#gas_pricing = pricing[pricing['exchangecode'].isin(["HHD", "PIG", "SCS"])]

#Load in x number of days worth of data
#folder_path = r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\2024 DAM Bid Data.zip"
#files = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) 

x = 365 
files_to_load = files[:x]  

daily_dfs = [pd.read_csv(file) for file in files_to_load]
one_day = pd.concat(daily_dfs, ignore_index=True)

#This is the dataframe with the solar identified and some of the generators identified
#confirmed_assets = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\resource_xrefs.csv")
confirmed_assets = pd.read_csv(RESOURCE_XREF_FILE)
confirmed_assets = confirmed_assets[confirmed_assets['gen_type'] != 'SOLAR']
confirmed_assets['asset_xref'] = confirmed_assets['resource_xref']

asset_day = pd.merge(asset_type, one_day, on='asset_xref', how='inner')
asset_day = pd.merge(asset_day, confirmed_assets, on='asset_xref', how='inner')
asset_day = asset_day[asset_day['resource_type'] == 'GENERATOR']
asset_day = asset_day.dropna(subset=['dyn_datetime_int'])

#Sort out the Hour-Ending Aspect Based on Chat Feedback
asset_day['dyn_datetime_int'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(asset_day['dyn_datetime_int'], unit='h')
asset_day['hour_ending'] = asset_day['dyn_datetime_int'].dt.hour
asset_day['hour_ending'] = asset_day['hour_ending'] + 1
asset_day = asset_day.drop(['schedulingcoordinator_seq', 'mineohstateofcharge', 'maxeohstateofcharge', 'dyn_datetime_int', 'resourcebid_seq'], axis=1)

#Filter gas prices for PG&E-Citygate for NP15, Socal-Citygate for SP15
gas_pricing = gas_pricing[gas_pricing['nodename'].isin(['PG&E-Citygate', 'Socal-Citygate'])]


import numpy as np

# --- 1) Filter asset_day to generators bidding energy ---
ad = asset_day.copy()
ad['opr_date'] = pd.to_datetime(ad['opr_date'])
ad = ad[
    (ad['resource_type'] == 'GENERATOR') &
    (ad['marketproducttype'] == 'EN')
].copy()

ad = ad[ad['quantity'].fillna(0) > 0]  # drop zero-quantity tranches

# Take lowest bid price per asset per hour
ad_min = (
    ad.groupby(['asset_xref', 'opr_date'], as_index=False)
      .agg({'price1': 'min'})
      .rename(columns={'price1': 'min_price_$per_MWh'})
)
ad_min['date'] = ad_min['opr_date'].dt.floor('D')

# --- 2) Convert gas_pricing wide -> long ---
gp = gas_pricing.copy()
gp['opr_month'] = pd.to_datetime(gp['opr_month'])
d_cols = [c for c in gp.columns if c.lower().startswith('d') and c[1:].isdigit()]

gp_long = gp.melt(
    id_vars=['opr_month', 'nodename'],
    value_vars=d_cols,
    var_name='dcol',
    value_name='gas_price_$per_MMBtu'
)
gp_long['day'] = gp_long['dcol'].str.extract(r'd0*([1-9]|[12]\d|3[01])').astype(float).astype('Int64')
gp_long['date'] = gp_long['opr_month'] + pd.to_timedelta(gp_long['day'] - 1, unit='D')
gp_long = gp_long[['date', 'nodename', 'gas_price_$per_MMBtu']].dropna(subset=['gas_price_$per_MMBtu'])

# --- 3) Split gas price by hub ---
gp_socal = gp_long[gp_long['nodename'].str.contains('Socal', case=False)].rename(
    columns={'gas_price_$per_MMBtu': 'gas_price_socal'}
)[['date', 'gas_price_socal']]

gp_pge = gp_long[gp_long['nodename'].str.contains('PG', case=False)].rename(
    columns={'gas_price_$per_MMBtu': 'gas_price_pge'}
)[['date', 'gas_price_pge']]

# --- 4) Merge both hub prices to asset data ---
merged = (
    ad_min
    .merge(gp_socal, on='date', how='left')
    .merge(gp_pge, on='date', how='left')
)

# --- 5) Compute both heatrates ---
merged['heatrate_socal'] = merged['min_price_$per_MWh'] / merged['gas_price_socal']
merged['heatrate_pge'] = merged['min_price_$per_MWh'] / merged['gas_price_pge']

# --- 6) Filter for heatrates within 5–15 for each hub ---
thermal_obs_socal = merged[(merged['heatrate_socal'] >= 5) & (merged['heatrate_socal'] <= 15)].copy()
thermal_obs_pge   = merged[(merged['heatrate_pge'] >= 5) & (merged['heatrate_pge'] <= 15)].copy()

# --- 7) Summaries per asset for each hub ---
def summarize(df, hr_col):
    return (
        df.groupby('asset_xref')
          .agg(
              total_hours=(hr_col, 'count'),
              hours_in_range=(hr_col, lambda x: x.between(5,20).sum()),
              pct_in_range=(hr_col, lambda x: x.between(5,20).mean()),
              median_hr=(hr_col, 'median'),
              std_hr=(hr_col, 'std'),
              q25_hr=(hr_col, lambda x: x.quantile(0.25)),
              q75_hr=(hr_col, lambda x: x.quantile(0.75))
          )
          .reset_index()
    )

sum_socal = summarize(merged, 'heatrate_socal').add_suffix('_socal')
sum_pge   = summarize(merged, 'heatrate_pge').add_suffix('_pge')

# Merge the two summaries
summary = sum_socal.merge(sum_pge, left_on='asset_xref_socal', right_on='asset_xref_pge', how='outer')

# Clean up column names
summary = summary.rename(columns={
    'asset_xref_socal': 'asset_xref'
}).drop(columns=['asset_xref_pge'])

# Thermal flags
summary['is_thermal_socal'] = (summary['pct_in_range_socal'] >= 0.5) & (summary['total_hours_socal'] >= 24)
summary['is_thermal_pge']   = (summary['pct_in_range_pge'] >= 0.5) & (summary['total_hours_pge'] >= 24)

# --- 8) Export results ---
# thermal_obs_socal.to_csv('thermal_obs_socal.csv', index=False)
# thermal_obs_pge.to_csv('thermal_obs_pge.csv', index=False)
# summary.to_csv('asset_heatrate_summary_dualhub.csv', index=False)

# print("Wrote: thermal_obs_socal.csv, thermal_obs_pge.csv, and asset_heatrate_summary_dualhub.csv")
# print(f"Socal thermal obs: {len(thermal_obs_socal)}, PGE thermal obs: {len(thermal_obs_pge)}")



