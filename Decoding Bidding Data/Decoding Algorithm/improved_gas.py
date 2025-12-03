import pandas as pd
import datetime as dt
import os
import glob

asset_type = pd.read_csv(r"C:\Users\harvi\OneDrive\Desktop\asset_xrefs.csv")

renewables = pd.read_csv(r"C:\Users\harvi\OneDrive\Desktop\renewables_2023.csv")

pricing = pd.read_csv(r"C:\Users\harvi\OneDrive\Desktop\energy_gas_prices_2023.csv")
electricty_pricing = pricing[pricing['exchangecode'].isin(["SQP", "DPN", "SDP", "UNP"])]
gas_pricing = pricing[pricing['exchangecode'].isin(["HHD", "PIG", "SCS"])]

hourly_pricing = pd.read_csv(r"C:\Users\harvi\OneDrive\Desktop\_SELECT_FROM_caiso_lmp_dam_AS_t1_INNER_JOIN_caiso_pnodes_xref_AS_202509290833.csv")


#Load in x number of days worth of data
folder_path = r"C:\Users\harvi\OneDrive\Desktop\2023 DAM Bid Data"
files = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) 

x = 100 
files_to_load = files[:x]  

daily_dfs = [pd.read_csv(file) for file in files_to_load]
one_day = pd.concat(daily_dfs, ignore_index=True)

#This is the dataframe with the solar identified and some of the generators identified
confirmed_assets = pd.read_csv(r"C:\Users\harvi\OneDrive\Desktop\resource_xrefs.csv")
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

# # Gettinh Heat Rates Included (Relationship Between Gas Pricing and Power Pricing to See What Gens Turn On)
# asset_day['SP HL heat rate'] = asset_day['SDP']/asset_day['SCS']
# asset_day['SP LL heat rate'] = asset_day['SQP']/asset_day['SCS']
# asset_day['NP HL heat rate'] = asset_day['DPN']/asset_day['PIG']
# asset_day['NP LL heat rate'] = asset_day['UNP']/asset_day['PIG']
# asset_day['SP HL calc heat rate'] = asset_day['price1']/asset_day['SCS']
# asset_day['SP LL calc heat rate'] = asset_day['price1']/asset_day['SCS']
# asset_day['NP HL calc heat rate'] = asset_day['price1']/asset_day['PIG']
# asset_day['NP LL calc heat rate'] = asset_day['price1']/asset_day['PIG']

# HOURLY SP15 HUB (Southern Cali) & NP15 HUB (Northern California) Pricing Inclusion Here

prices_long = hourly_pricing.melt(
    id_vars=['opr_date', 'node_id'],
    value_vars=[f'he{i:02d}' for i in range(1, 25)],
    var_name='hour_col',
    value_name='price'
)

prices_long['hour_ending'] = prices_long['hour_col'].str.extract(r'he(\d+)').astype(int)


prices_pivot = prices_long.pivot_table(
    index=['opr_date','hour_ending'],
    columns='node_id',
    values='price'
).reset_index()


prices_pivot['opr_date'] = pd.to_datetime(prices_pivot['opr_date']).dt.date

bids_with_prices = pd.merge(asset_day,
    prices_pivot,
    on=['opr_date','hour_ending']
)


bids_with_prices['NP15_Hourly_Price'] = bids_with_prices['TH_NP15_GEN-APND']
bids_with_prices['SP15_Hourly_Price'] = bids_with_prices['TH_SP15_GEN-APND']

bids_with_prices = bids_with_prices.drop(['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND'], axis=1)


bids_with_prices = bids_with_prices[~bids_with_prices['asset_xref'].isin([1000457, 1001121, 1000440])] #REMOVE CONFIRMED NUKES

#Remove the self-scheduled stuff
bids_with_prices = bids_with_prices[~(bids_with_prices['selfschedmw'] > 0)]
bids_with_prices = bids_with_prices[~(bids_with_prices['selfschedmw'] <= 0)]
print(bids_with_prices.head())


import numpy as np
import pandas as pd

import numpy as np

def classify_gas_confidence(df):
    results = []

    for xref, group in df.groupby('asset_xref'):
        # Skip assets with too few hours
        if len(group) < 50:
            continue
        
        # --- 1. Correlation with Power Price ---
        price_corr = group['quantity'].corr(group['SP15_Hourly_Price'])
        price_corr = np.clip(price_corr, 0, 1) if not np.isnan(price_corr) else 0
        
        # --- 2. Correlation with Heat Rate ---
        heat_rate = group['SP15_Hourly_Price'] / group['SCS']
        heat_corr = group['quantity'].corr(heat_rate)
        heat_corr = np.clip(heat_corr, 0, 1) if not np.isnan(heat_corr) else 0
        
        # --- 3. Evening Share ---
        evening_hours = group[group['hour_ending'].between(18, 22)]
        evening_share = evening_hours['quantity'].sum() / (group['quantity'].sum() + 1e-9)
        evening_share = np.clip(evening_share, 0, 1)
        
        # --- 4. Correlation with Solar Output ---
        solar_corr = group['quantity'].corr(group[['SP15_SOLAR', 'NP15_SOLAR']].sum(axis=1))
        solar_corr = np.clip(solar_corr, -1, 1)
        
        # --- 5. Cycling Score (On/Off frequency) ---
        onoff = (group['quantity'] > 10).astype(int)
        cycling_score = onoff.diff().abs().sum() / len(onoff)
        cycling_score = np.clip(cycling_score, 0, 1)
        
        
        # --- 6. Heat Rate x Price Correlation ---
        heat_rate = evening_hours['SP15_Hourly_Price'] / evening_hours['SCS']
        heat_corr2 = evening_hours['price1'].corr(heat_rate)
        heat_corr2 = np.clip(heat_corr2, 0, 1) if not np.isnan(heat_corr2) else 0
        
        # --- Final Confidence ---
        confidence = (
            0.2 * price_corr +
            0.25 * heat_corr +
            0.2 * evening_share +
            0.05 * (1 - abs(solar_corr)) + 0.2 * heat_corr2 +
            0.1 * cycling_score
        )
        
        results.append({
            'asset_xref': xref,
            'price_corr': price_corr,
            'heat_corr': heat_corr,
            'evening_share': evening_share,
            'solar_corr': solar_corr,
            'cycling_score': cycling_score,
            'price-heat_corr': heat_corr2,
            'gas_confidence': confidence
        })
    
    return pd.DataFrame(results)


gas_scores = classify_gas_confidence(bids_with_prices)
