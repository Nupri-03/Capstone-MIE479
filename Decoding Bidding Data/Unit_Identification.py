import pandas as pd
import datetime as dt
import os
import glob

asset_type = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\asset_xrefs.csv")
gen_meta_data = pd.read_csv(
    r"C:\Users\jliu\Downloads\GEN_META_CAPSTONE.csv",
    encoding="latin-1"
)

#Remove units that did not generate in 2024
months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

gen_meta_data = gen_meta_data[~(gen_meta_data[months].eq(0).all(axis=1))]

# Keep only rows whose month values appear more than once
matching_rows = gen_meta_data[gen_meta_data.duplicated(subset=months, keep=False)]


#Load in x number of days worth of data
folder_path = r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\2024 DAM Bid Data"
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

#Find max bid per month based on bid data
asset_day['bid_mw'] = asset_day['selfschedmw'].fillna(0)

mask = asset_day['bid_mw'] == 0
asset_day.loc[mask, 'bid_mw'] = asset_day.loc[mask, 'quantity']

asset_day['opr_date'] = pd.to_datetime(asset_day['opr_date'])
asset_day['month'] = asset_day['opr_date'].dt.month

monthly_max = (
    asset_day
    .groupby(['asset_xref', 'month'])['bid_mw']
    .max()
    .reset_index()
)

#Match gen_meta_data with max monthly bids
monthly_pivot = (
    monthly_max
    .pivot(index='asset_xref', columns='month', values='bid_mw')
    .reset_index()
)

month_map = {
    1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR',
    5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG',
    9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
}

monthly_pivot = monthly_pivot.rename(columns=month_map)

months = ["JAN","FEB","MAR","APR","MAY","JUN",
          "JUL","AUG","SEP","OCT","NOV","DEC"]


# --- Step 2: Add meta_id to gen_meta_data for unique identification ---
gen_meta_data = gen_meta_data.reset_index().rename(columns={'index':'meta_id'})


# --- Step 3: Perform full merge (one asset_xref may match multiple rows) ---
merged = monthly_pivot.merge(gen_meta_data, on=months, how='inner')


# --- Step 4: Count how many matches each asset_xref got ---
match_counts = merged.groupby('asset_xref').size().reset_index(name='n_matches')

# Assets with >1 matches
ambiguous_assets = match_counts[match_counts['n_matches'] > 1]['asset_xref']

# Assets with exactly 1 match
unique_assets = match_counts[match_counts['n_matches'] == 1]['asset_xref']


# --- Step 5: Build ambiguous and final unique match DataFrames ---

# All ambiguous rows (for manual review)
ambiguous_matches = merged[merged['asset_xref'].isin(ambiguous_assets)].copy()

# The safe one-to-one matches
final_unique_matches = merged[merged['asset_xref'].isin(unique_assets)].copy()

# Ensure exactly 1 row per asset_xref (since these are safe)
final_unique_matches = final_unique_matches.drop_duplicates(subset=['asset_xref'])

gen_meta_data['asset_xref'] = ""
gen_meta_with_xref = gen_meta_data.merge(
    final_unique_matches[['meta_id', 'asset_xref']],
    on='meta_id',
    how='left',
    suffixes=('', '_match')
)

months = ["JAN","FEB","MAR","APR","MAY","JUN",
          "JUL","AUG","SEP","OCT","NOV","DEC"]

# Ensure all month columns exist
for m in months:
    if m not in gen_meta_with_xref.columns:
        gen_meta_with_xref[m] = 0

# Compute annual max per row
gen_meta_with_xref['annual_max'] = gen_meta_with_xref[months].max(axis=1)

# Keep only rows where annual_max >= 5
gen_meta_with_xref = gen_meta_with_xref[gen_meta_with_xref['annual_max'] >= 5].copy()

# Optionally, drop the helper column if you don’t need it
gen_meta_with_xref = gen_meta_with_xref.drop(columns=['annual_max'])

gen_meta_with_xref.to_excel(
    "gen_meta_with_xref.xlsx",  # filename
    index=False,                # don’t include the DataFrame index
    engine='openpyxl'           # use openpyxl engine
)


public_queue_report = pd.read_csv(
    r"C:\Users\jliu\Downloads\publicqueuereport.csv",
    encoding="latin-1"
)

# Unmatched assets
unmatched_assets = monthly_pivot[
    ~monthly_pivot['asset_xref'].isin(gen_meta_with_xref['asset_xref'])
].copy()

months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

# Ensure all month columns exist
for m in months:
    if m not in unmatched_assets.columns:
        unmatched_assets[m] = 0

# Compute the annual max for each unmatched asset
unmatched_assets['annual_max'] = unmatched_assets[months].max(axis=1)

# Extract relevant columns
pq_subset = public_queue_report.iloc[:, [0, 15]].copy()
pq_subset.columns = ['gen_name', 'pq_max_gen']

# Ensure numeric type
pq_subset['pq_max_gen'] = pd.to_numeric(pq_subset['pq_max_gen'], errors='coerce')
# Remove rows where all month columns are NaN
unmatched_assets = unmatched_assets.dropna(subset=months, how='all')


import numpy as np

matches = []

# Step 1: Compute all differences
for _, asset in unmatched_assets.iterrows():
    diffs = np.abs(pq_subset['pq_max_gen'] - asset['annual_max'])
    for idx, diff in diffs.items():
        matches.append({
            'asset_xref': asset['asset_xref'],
            'matched_gen_name': pq_subset.loc[idx, 'gen_name'],
            'annual_max': asset['annual_max'],
            'pq_max_gen': pq_subset.loc[idx, 'pq_max_gen'],
            'difference': diff
        })

public_queue_matches = pd.DataFrame(matches)

# Step 2: Compute relative difference
public_queue_matches['relative_diff'] = public_queue_matches['difference'] / public_queue_matches['pq_max_gen']

# Step 3: Sort by relative difference (smallest first)
public_queue_matches = public_queue_matches.sort_values('relative_diff')

# Step 4: Greedy one-to-one assignment
assigned_assets = set()
assigned_gens = set()
final_assignments = []

tolerance = 0.05  # 5% relative difference

for _, row in public_queue_matches.iterrows():
    asset = row['asset_xref']
    gen = row['matched_gen_name']
    
    if asset not in assigned_assets and gen not in assigned_gens and row['relative_diff'] <= tolerance:
        final_assignments.append(row)
        assigned_assets.add(asset)
        assigned_gens.add(gen)

# Step 5: Create final DataFrame
strong_matches_greedy = pd.DataFrame(final_assignments)

# Ensure asset_xref is string
strong_matches_greedy['asset_xref'] = strong_matches_greedy['asset_xref'].astype(str)

# Remove any trailing .0 or decimals
strong_matches_greedy['asset_xref'] = strong_matches_greedy['asset_xref'].str.replace(r'\.0+$', '', regex=True)
strong_matches_greedy['asset_xref'] = strong_matches_greedy['asset_xref'].str.replace(r'\.\d+$', '', regex=True)

# Compute MW: either self-scheduled or sum of traunches
asset_day['mw_bid'] = asset_day.apply(
    lambda row: row['selfschedmw'] if row['selfschedmw'] > 0 else row['quantity'],
    axis=1
)

max_bid_per_asset = (
    asset_day.groupby('asset_xref')['mw_bid']
    .max()
    .reset_index()
    .rename(columns={'mw_bid': 'max_mw_bid'})
)

max_bid_overall = asset_day['mw_bid'].max()

# Step 1: merge monthly and annual max
merged = monthly_max.merge(
    max_bid_per_asset,
    on="asset_xref",
    how="left"
)

# Step 2: pivot to make months columns
wide = merged.pivot(
    index="asset_xref",
    columns="month",
    values="bid_mw"
)

# Step 3: add annual max back in
wide = wide.merge(
    max_bid_per_asset,
    on="asset_xref",
    how="left"
)





