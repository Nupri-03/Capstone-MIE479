import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import glob

# ============================================================
# === LOAD DATA ==============================================
# ============================================================

asset_type = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\asset_xrefs.csv")

folder_path = r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\2024 DAM Bid Data"
files = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) 

x = 365  # number of daily files to load
files_to_load = files[:x]  

daily_dfs = [pd.read_csv(file) for file in files_to_load]
one_day = pd.concat(daily_dfs, ignore_index=True)

asset_day = pd.merge(asset_type, one_day, on='asset_xref', how='inner')
asset_day = asset_day[asset_day['resource_type'] == 'GENERATOR']
asset_day = asset_day.dropna(subset=['dyn_datetime_int'])

# Convert dyn_datetime_int into hour_ending
asset_day['dyn_datetime_int'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(asset_day['dyn_datetime_int'], unit='h')
asset_day['hour_ending'] = asset_day['dyn_datetime_int'].dt.hour + 1
asset_day = asset_day.drop(['schedulingcoordinator_seq', 'mineohstateofcharge', 
                            'maxeohstateofcharge', 'dyn_datetime_int', 'resourcebid_seq'], axis=1)

filtered_bids = asset_day[asset_day['asset_xref'] == 1000676]
filtered_bids = filtered_bids[filtered_bids['marketproducttype'] == 'EN']

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Step 1: Clean the data ---
# We only use MW (quantity) and price (price1)
bids = filtered_bids.copy()

# Convert to numeric (in case some rows contain strings)
bids['quantity'] = pd.to_numeric(bids['quantity'], errors='coerce')
bids['price1']   = pd.to_numeric(bids['price1'], errors='coerce')

# Remove rows missing MW or price
bids = bids.dropna(subset=['quantity', 'price1'])

# Feature matrix
X = bids[['quantity', 'price1']].values

# --- Step 2: Standardize features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 3: Run KMeans Clustering ---
# 4 clusters works well for HDPP: CT1, CT2, CT+ST, CCGT
# kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
# clusters = kmeans.fit_predict(X_scaled)

# # Add cluster labels to df
# bids['cluster'] = clusters

# # --- Step 4: Plot ---
# plt.figure(figsize=(10,6))
# plt.scatter(bids['quantity'], bids['price1'], c=bids['cluster'], cmap='tab10', s=80)
# plt.xlabel("Quantity (MW)")
# plt.ylabel("Price ($/MWh)")
# plt.title("Cluster Analysis of Bid Tiers for Asset 1000676")
# plt.grid(True)
# plt.show()

# # --- Step 5: Summary of clusters ---
# summary = bids.groupby('cluster')['quantity'].agg(['count','min','max','mean']).sort_values('mean')
# print("\nCluster Summary Based on MW:\n")
# print(summary)

# summary2 = bids.groupby('cluster')['price1'].agg(['min','max','mean'])
# print("\nCluster Price Summary:\n")
# print(summary2)

pricing = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\Energy_Gas_Prices_2024.csv")
electricty_pricing = pricing[pricing['exchangecode'].isin(["SQP", "DPN", "SDP", "UNP"])]
gas_pricing = pricing[pricing['exchangecode'].isin(["HHD", "PIG", "SCS"])]

load_forecast = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\load_forecasts_2024.csv")
load_forecast = load_forecast[load_forecast['nodename'].isin(['CA ISO-TAC'])]

import re

load = load_forecast.copy()
load['opr_date'] = pd.to_datetime(load['opr_date'])

# --- Identify only columns that look like hour columns (he01–he24) ---
hour_cols = [c for c in load.columns if re.match(r'he\d+', c.lower())]

# --- Melt only the correct hour columns ---
load_long = (
    load.melt(
        id_vars=['opr_date','nodename'],
        value_vars=hour_cols,
        var_name='hour',
        value_name='load_mw'
    )
)

# Extract hour number safely
load_long['hour'] = load_long['hour'].str.extract(r'(\d+)')
load_long['hour'] = pd.to_numeric(load_long['hour'], errors='coerce')

# Now drop rows where hour extraction failed
load_long = load_long.dropna(subset=['hour'])

# Convert to int
load_long['hour'] = load_long['hour'].astype(int)

# Keep CA ISO-TAC
load_long = load_long[load_long['nodename'] == 'CA ISO-TAC']
load_long = load_long.drop(columns=['nodename'])

gas = gas_pricing.copy()
gas['opr_month'] = pd.to_datetime(gas['opr_month'])

# --- Filter to Socal-Citygate only ---
gas = gas[gas['nodename'] == 'Socal-Citygate']

# --- Identify only columns that look like daily price columns (d01–d31) ---
day_cols = [c for c in gas.columns if re.match(r'd\d+', c.lower())]

# --- Melt only the correct day columns ---
gas_long = (
    gas.melt(
        id_vars=['opr_month', 'nodename', 'ice_xref'],
        value_vars=day_cols,
        var_name='day',
        value_name='gas_price'
    )
)

# Extract the day number safely
gas_long['day'] = gas_long['day'].str.extract(r'(\d+)')
gas_long['day'] = pd.to_numeric(gas_long['day'], errors='coerce')

# Drop rows where day extraction failed
gas_long = gas_long.dropna(subset=['day'])

# Convert to int
gas_long['day'] = gas_long['day'].astype(int)

# Build real opr_date
gas_long['opr_date'] = gas_long['opr_month'] + pd.to_timedelta(gas_long['day'] - 1, unit='D')

# Keep only needed columns
gas_long = gas_long[['opr_date', 'gas_price']]

# Fix dtypes on bids
bids['opr_date'] = pd.to_datetime(bids['opr_date'])
bids['hour_ending'] = pd.to_numeric(bids['hour_ending'], errors='coerce').astype('int64')

# Fix dtypes on load_long
load_long['opr_date'] = pd.to_datetime(load_long['opr_date'])
load_long['hour'] = pd.to_numeric(load_long['hour'], errors='coerce').astype('int64')


# merge bids with load
df = bids.merge(load_long, 
                left_on=['opr_date','hour_ending'],
                right_on=['opr_date','hour'],
                how='left')

df = df.drop(columns=['hour'])

# keep only economic bids
df = df[df['price1'] < 900].copy()

# merge bids+load with gas prices
df = df.merge(gas_long, on='opr_date', how='left')

# remove missing rows
df = df.dropna(subset=['load_mw','gas_price','price1'])

# Add time features
df['month'] = df['opr_date'].dt.month
df['dow']   = df['opr_date'].dt.dayofweek

# ============================================================
# === MODELLING WITH ADAPTIVE PER-CLUSTER IMPROVEMENT ========
# ============================================================

print("\n=== START MODELLING ===")

# ------------------------------------------------------------
# 1. CLUSTER MW TRANCHES (economic bids only)
# ------------------------------------------------------------

cluster_source = bids[bids['price1'] < 800].copy()
cluster_source = cluster_source.dropna(subset=['quantity', 'price1'])

cluster_source['quantity'] = pd.to_numeric(cluster_source['quantity'], errors='coerce')
cluster_source['price1']   = pd.to_numeric(cluster_source['price1'], errors='coerce')

# Use both MW + price
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_cluster = scaler.fit_transform(cluster_source[['quantity','price1']])

# Determine k with silhouette
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

K = range(5, 15)
sil_scores = []

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X_cluster)
    sil_scores.append(silhouette_score(X_cluster, labels))

k_opt = K[sil_scores.index(max(sil_scores))]
print(f"Selected optimal k = {k_opt}")

# Fit final KMeans
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init='auto')
cluster_source['cluster'] = kmeans.fit_predict(X_cluster)

# ------------------------------------------------------------
# 2. MERGE CLUSTER LABELS BACK INTO df
# ------------------------------------------------------------
df = df.merge(
    cluster_source[['opr_date','hour_ending','bid_tier','cluster']],
    on=['opr_date','hour_ending','bid_tier'],
    how='left'
)

print("\nCluster distribution:")
print(df['cluster'].value_counts().sort_index())

# ------------------------------------------------------------
# 3. TRAIN/VALIDATION SPLIT
# ------------------------------------------------------------

train = df[df['opr_date'] < '2024-11-01'].copy()
val   = df[df['opr_date'] >= '2024-11-01'].copy()

features = ['load_mw','gas_price','month','dow','hour_ending']

models = {}
cluster_errors = {}

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ------------------------------------------------------------
# 4. BASELINE MODELS (RandomForest)
# ------------------------------------------------------------

for c in sorted(df['cluster'].unique()):
    train_c = train[train['cluster'] == c]
    val_c   = val[val['cluster'] == c]

    if len(train_c) < 20 or len(val_c) < 5:
        print(f"Skipping cluster {c}: insufficient data")
        continue

    X_train = train_c[features]
    y_train = train_c['price1']

    X_val = val_c[features]
    y_val = val_c['price1']

    # Baseline RF model
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_split=4
    )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_val)
    mae = mean_absolute_error(y_val, pred)

    models[c] = rf
    cluster_errors[c] = mae

print("\n=== BASELINE (RANDOM FOREST) ERRORS ===")
for c,e in cluster_errors.items():
    print(f"Cluster {c}: MAE = {e:.2f}")

# ------------------------------------------------------------
# 5. IMPROVE BAD CLUSTERS (XGBoost fallback)
# ------------------------------------------------------------

from xgboost import XGBRegressor

improvement_threshold = np.percentile(list(cluster_errors.values()), 70)  
# try improving bottom 30%

for c, err in cluster_errors.items():

    if err < improvement_threshold:
        continue    # baseline model is good enough

    print(f"\nCluster {c} has high error ({err:.2f}), trying XGBoost...")

    train_c = train[train['cluster'] == c]
    val_c   = val[val['cluster'] == c]

    X_train = train_c[features]
    y_train = train_c['price1']
    X_val   = val_c[features]
    y_val   = val_c['price1']

    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective='reg:squarederror'
    )

    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_val)
    mae = mean_absolute_error(y_val, pred)

    if mae < cluster_errors[c]:
        print(f"--> Replaced RF with XGBoost for cluster {c}: MAE improved {cluster_errors[c]:.2f} → {mae:.2f}")
        models[c] = xgb
        cluster_errors[c] = mae
    else:
        print(f"--> XGBoost did NOT improve cluster {c} (MAE={mae:.2f}), keeping RF.")

# ------------------------------------------------------------
# 6. FINAL VALIDATION + SCATTER PLOTS
# ------------------------------------------------------------

val_results = val.copy()
pred_list = []

for idx, row in val_results.iterrows():
    c = row['cluster']
    if c in models:
        pred_list.append(models[c].predict(row[features].values.reshape(1,-1))[0])
    else:
        pred_list.append(np.nan)

val_results['predicted_price'] = pred_list
val_vis = val_results.dropna(subset=['predicted_price']).copy()

# PLOTS
pred_color = 'blue'

for c in sorted(val_vis['cluster'].unique()):
    
    cluster_slice = val_vis[val_vis['cluster'] == c]
    if len(cluster_slice) < 5:
        continue

    plt.figure(figsize=(7,6))
    plt.scatter(cluster_slice['price1'], cluster_slice['predicted_price'],
                color=pred_color, s=45, alpha=0.7, label='Predicted')

    mn = min(cluster_slice['price1'].min(), cluster_slice['predicted_price'].min())
    mx = max(cluster_slice['price1'].max(), cluster_slice['predicted_price'].max())
    plt.plot([mn, mx], [mn, mx], 'k--', label='Perfect Fit')

    plt.xlabel("Actual Price ($/MWh)")
    plt.ylabel("Predicted Price ($/MWh)")
    plt.title(f"Cluster {c} — Actual vs Predicted (Final Model)")
    plt.grid(True)
    plt.legend()
    plt.show()

import seaborn as sns

plt.figure(figsize=(8,5))
sns.kdeplot(val_vis['price1'], label='Actual', shade=True)
sns.kdeplot(val_vis['predicted_price'], label='Predicted', shade=True)
plt.title("Actual vs Predicted Price Distribution")
plt.legend()
plt.show()

from sklearn.inspection import PartialDependenceDisplay

for c,model in models.items():
    if len(train[train.cluster==c]) < 20: continue
    fig = PartialDependenceDisplay.from_estimator(
        model, 
        train[train['cluster']==c][features], 
        features=['load_mw',"gas_price"],
        n_cols=2
    )
    plt.suptitle(f"PDP — Cluster {c}")
    plt.show()




