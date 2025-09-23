# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import datetime as dt
import os
import glob

asset_type = pd.read_csv(r"C:\Users\harvi\OneDrive\Desktop\asset_xref_202509121109.csv")

renewables = pd.read_csv(r"C:\Users\harvi\OneDrive\Desktop\caiso_sld_ren_fcst_202509191057.csv")

pricing = pd.read_csv(r"C:\Users\harvi\OneDrive\Desktop\_SELECT_s_x_FROM_west_fin_ice_settlement_s_JOIN_west_fin_ice_xre_202509191043.csv")
electricty_pricing = pricing[pricing['exchangecode'].isin(["SQP", "DPN", "SDP", "UNP"])]
gas_pricing = pricing[pricing['exchangecode'].isin(["HHD", "PIG", "SCS"])]

#Load in x number of days worth of data
folder_path = r"C:\Users\harvi\OneDrive\Desktop\2023 DAM Bid Data"
files = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) 

x = 7 
files_to_load = files[:x]  

daily_dfs = [pd.read_csv(file) for file in files_to_load]
one_day = pd.concat(daily_dfs, ignore_index=True)

#This is the dataframe with the solar identified and some of the generators identified
confirmed_assets = pd.read_csv(r"C:\Users\harvi\OneDrive\Desktop\resource_xref_202509121107.csv")
confirmed_assets = confirmed_assets[confirmed_assets['gen_type'] != 'SOLAR']
confirmed_assets['asset_xref'] = confirmed_assets['resource_xref']

asset_day = pd.merge(asset_type, one_day, on='asset_xref', how='inner')
asset_day = pd.merge(asset_day, confirmed_assets, on='asset_xref', how='inner')
asset_day = asset_day[asset_day['resource_type'] == 'GENERATOR']
asset_day = asset_day.dropna(subset=['dyn_datetime_int'])

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


#Convert the Electricty Pricing Into Something Useable
value_vars = [col for col in electricty_pricing.columns if col.startswith('d')]
df_long = electricty_pricing.melt(
    id_vars=['opr_month', 'exchangecode'],
    value_vars=value_vars,
    var_name='hour_ending',
    value_name='price'
)

df_long = df_long.dropna()

df_long['opr_date'] = pd.to_datetime(df_long['opr_month']) + pd.to_timedelta(
    df_long['hour_ending'].str.extract('(\d+)')[0].astype(int) - 1, unit='D'
)

df_pivot = df_long.pivot_table(
    index=['opr_date'],
    columns='exchangecode',
    values='price'
).reset_index()

df_pivot['opr_date'] = pd.to_datetime(df_pivot['opr_date']).dt.date
asset_day['opr_date'] = pd.to_datetime(asset_day['opr_date']).dt.date
renew_pivot['opr_date'] = pd.to_datetime(renew_pivot['opr_date']).dt.date

asset_day = pd.merge(asset_day, df_pivot, on='opr_date')




#Convert the Gas Pricing Into Something Useable
value_vars = [col for col in gas_pricing.columns if col.startswith('d')]
df_long = gas_pricing.melt(
    id_vars=['opr_month', 'exchangecode'],
    value_vars=value_vars,
    var_name='hour_ending',
    value_name='price'
)

df_long = df_long.dropna()

df_long['opr_date'] = pd.to_datetime(df_long['opr_month']) + pd.to_timedelta(
    df_long['hour_ending'].str.extract('(\d+)')[0].astype(int) - 1, unit='D'
)

df_pivot = df_long.pivot_table(
    index=['opr_date'],
    columns='exchangecode',
    values='price'
).reset_index()

df_pivot['opr_date'] = pd.to_datetime(df_pivot['opr_date']).dt.date

asset_day = pd.merge(asset_day, df_pivot, on='opr_date')

#Sort out the Hour-Ending Aspect Based on Chat Feedback
asset_day['dyn_datetime_int'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(asset_day['dyn_datetime_int'], unit='h')
asset_day['hour_ending'] = asset_day['dyn_datetime_int'].dt.hour
asset_day['hour_ending'] = asset_day['hour_ending'] + 1
asset_day = asset_day.drop(['schedulingcoordinator_seq', 'mineohstateofcharge', 'maxeohstateofcharge', 'dyn_datetime_int', 'resourcebid_seq'], axis=1)

asset_day = pd.merge(asset_day, renew_pivot, on=['opr_date', 'hour_ending'])

#Remove the self-scheduled stuff

asset_day = asset_day[~(asset_day['selfschedmw'] > 0)]
asset_day = asset_day[~(asset_day['selfschedmw'] <= 0)]
print(asset_day.head())
