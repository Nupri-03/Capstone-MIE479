# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 17:21:13 2025

@author: karathah
"""

# data_prep_for_bucketing.py
# Requirements: pandas, numpy, tqdm, xgboost, scikit-learn
# Run in Colab or local. Adjust paths to your zip/folders.

import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from tqdm import tqdm
import re
import warnings
from io import TextIOWrapper
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------- USER PATHS / SETTINGS ----------
BIDS_ZIP_OR_FOLDER = r"\\VSRV2\C.Homes$\karathah\Desktop\Data Required for CAPSTONE Modelling\2024 DAM Bid Data.zip"
LOAD_CSV =  r"\\VSRV2\C.Homes$\karathah\Desktop\Data Required for CAPSTONE Modelling\load_tesla_fcst_202511162319.csv"
POWER_PRICING_CSV = r"\\VSRV2\C.Homes$\karathah\Desktop\Data Required for CAPSTONE Modelling\power_pricing.csv"
GAS_PRICES_CSV =  r"\\VSRV2\C.Homes$\karathah\Desktop\Data Required for CAPSTONE Modelling\Gas_Prices_2024.csv"  # optional
RENEWABLES_CSV = r"\\VSRV2\C.Homes$\karathah\Desktop\Data Required for CAPSTONE Modelling\Renewables_2024.csv"
OUTPUT_MODELING_CSV = "modeling_dataframe.csv"

# Buckets: $0–5, 5–10, … 200
BUCKET_STEP = 5
BUCKET_MAX = 200
BUCKET_EDGES = list(range(-25, BUCKET_MAX + BUCKET_STEP, BUCKET_STEP))

SP_NODE_ID_FRAGMENT = "SP15"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def load_all_bids(path):
    """Load all bid_YYYYMMDD_*.csv from ZIP or folder."""
    all_dfs = []
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, 'r') as z:
            csvs = [f for f in z.namelist() if f.lower().endswith(".csv")]
            for fname in csvs:
                m = re.search(r"bid_(\d{8})_", fname)
                if not m:
                    continue
                d = pd.to_datetime(m.group(1), format="%Y%m%d")
                with z.open(fname) as f:
                    df = pd.read_csv(TextIOWrapper(f, "utf-8"), low_memory=False)
                df["opr_date_from_filename"] = d
                all_dfs.append(df)
    else:
        import os
        for fname in os.listdir(path):
            if not fname.lower().endswith(".csv"):
                continue
            m = re.search(r"bid_(\d{8})_", fname)
            if not m:
                continue
            d = pd.to_datetime(m.group(1), format="%Y%m%d")
            df = pd.read_csv(os.path.join(path, fname), low_memory=False)
            df["opr_date_from_filename"] = d
            all_dfs.append(df)
    if not all_dfs:
        raise ValueError("No valid bid files found.")
    return pd.concat(all_dfs, ignore_index=True)

def find_price_columns(df):
    return [c for c in df.columns if re.match(r'(?i)^price', c)]

def collapse_price_columns(df, price_cols):
    df["price"] = np.nan
    for c in price_cols:
        df["price"] = df["price"].fillna(pd.to_numeric(df[c], errors="coerce"))
    return df


# ------------------------------------------------------------
# 1) LOAD BIDS
# ------------------------------------------------------------
bids = load_all_bids(BIDS_ZIP_OR_FOLDER)
bids.columns = [c.strip() for c in bids.columns]

# Quantity column detection
quantity_col = None
for qname in ["quantity","mw","quantity_mw","bidquantity","quantity1"]:
    if qname in bids.columns:
        quantity_col = qname
        break
if quantity_col is None:
    raise ValueError("Could not detect quantity column.")

bids["quantity"] = pd.to_numeric(bids[quantity_col], errors="coerce").fillna(0)

# Prices
price_cols = find_price_columns(bids)
bids = collapse_price_columns(bids, price_cols)

# Date + hour

bids["date"] = pd.to_datetime(bids["opr_date_from_filename"])
bids['dyn_datetime_int'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(bids['dyn_datetime_int'], unit='h')
bids['hour'] = bids['dyn_datetime_int'].dt.hour
bids['hour'] = bids['hour'] + 1
bids = bids.drop(['mineohstateofcharge', 'maxeohstateofcharge', 'dyn_datetime_int'], axis=1)

bids.loc[bids["hour"] < 1, "hour"] = 1
bids.loc[bids["hour"] > 24, "hour"] = 24

# Keep essentials
b = bids[["date","hour","price","quantity"]].copy()
b["quantity"] = pd.to_numeric(b["quantity"], errors="coerce").fillna(0)
b = b[b["quantity"] > 0].reset_index(drop=True)

# ------------------------------------------------------------
# 2) HOURLY SUPPLY CURVES
# ------------------------------------------------------------
def hourly_supply(df):
    df = df.sort_values("price").copy()
    df["cum_mw"] = df["quantity"].cumsum()
    return df[["price","cum_mw"]]

supply_map = {}
for (d,h), dfh in tqdm(b.groupby(["date","hour"]), desc="Hourly curves"):
    supply_map[(d,h)] = hourly_supply(dfh).to_numpy()

# ------------------------------------------------------------
# 3) BUCKET MW
# ------------------------------------------------------------
def cum_mw_at_price(arr, p):
    if arr is None or len(arr) == 0:
        return 0.0
    idx = np.searchsorted(arr[:,0], p, side="right") - 1
    if idx < 0:
        return 0.0
    return float(arr[idx,1])

bucket_rows = []
for (d,h), arr in tqdm(supply_map.items(), desc="Bucketizing"):
    for i in range(len(BUCKET_EDGES)-1):
        lo = BUCKET_EDGES[i]
        hi = BUCKET_EDGES[i+1]
        mw_hi = cum_mw_at_price(arr, hi)
        mw_lo = cum_mw_at_price(arr, lo)
        bucket_rows.append({
            "date": d,
            "hour": h,
            "bucket_low": lo,
            "bucket_high": hi,
            "bucket_label": f"${lo}-{hi}",
            "mw_in_bucket": max(0, mw_hi - mw_lo)
        })
buckets_df = pd.DataFrame(bucket_rows)

# ------------------------------------------------------------
# 4) LOAD TESLA/CAISO LOAD
# ------------------------------------------------------------
load = pd.read_csv(LOAD_CSV)
load.columns = [c.strip() for c in load.columns]
load["date"] = pd.to_datetime(load["opr_date"])

he_cols = [c for c in load.columns if re.match(r"(?i)^he0?\d", c)]
load_long = load.melt(id_vars=["date","nodename"], value_vars=he_cols,
                      var_name="he", value_name="load_mw")
load_long["hour"] = load_long["he"].str.extract(r"(\d+)$").astype(int)
load_long = load_long[["date","hour","nodename","load_mw"]]

# ------------------------------------------------------------
# 5) LOAD SP15 PRICES
# ------------------------------------------------------------
pp = pd.read_csv(POWER_PRICING_CSV)
pp.columns = [c.strip() for c in pp.columns]
pp["date"] = pd.to_datetime(pp["opr_date"])

he_cols = [c for c in pp.columns if re.match(r"(?i)^he0?\d", c)]
sp_long = pp.melt(id_vars=["date","node_id"], value_vars=he_cols,
                  var_name="he", value_name="price")
sp_long["hour"] = sp_long["he"].str.extract(r"(\d+)$").astype(int)
sp_long = sp_long[["date","hour","price","node_id"]]

# ------------------------------------------------------------
# 6) LOAD RENEWABLES
# ------------------------------------------------------------
renew = pd.read_csv(RENEWABLES_CSV, low_memory=False)
renew.columns = [c.strip() for c in renew.columns]
renew['date'] = pd.to_datetime(renew['opr_date'], errors='coerce')

renew_he_cols = [c for c in renew.columns if re.match(r'(?i)^he0?\d', c)]
renew_long = renew.melt(id_vars=['date','trading_hub','renewable_type','market_run_id'],
                        value_vars=renew_he_cols, var_name='he', value_name='mw')
renew_long['hour'] = renew_long['he'].str.extract(r'(\d+)$').astype(int)
renew_long['mw'] = pd.to_numeric(renew_long['mw'], errors='coerce').fillna(0)

renew_agg = (renew_long
             .groupby(['date','hour','trading_hub','renewable_type'])['mw']
             .sum()
             .reset_index())

renew_pivot = (renew_agg
               .pivot_table(index=['date','hour','trading_hub'],
                            columns='renewable_type', values='mw', aggfunc='sum', fill_value=0)
               .reset_index())
renew_pivot.columns = [str(c).lower() if not isinstance(c, tuple) else c for c in renew_pivot.columns]
renew_pivot = renew_pivot.rename(columns=lambda x: str(x).lower())
for c in list(renew_pivot.columns):
    if c not in ['date','hour','trading_hub']:
        renew_pivot = renew_pivot.rename(columns={c: f"{c.lower()}_mw"})

renew_sp = renew_pivot[renew_pivot['trading_hub'].str.upper() == 'SP15'].copy()
renew_sp = renew_sp[['date','hour'] + [c for c in renew_sp.columns if c.endswith('_mw')]]

# ------------------------------------------------------------
# 7) MERGE BUCKETS, LOAD, PRICES, RENEWABLES
# ------------------------------------------------------------
# Ensure all hour columns are int
for df_hour in [buckets_df, load_long, sp_long, renew_sp]:
    df_hour['hour'] = df_hour['hour'].astype(int)

# Merge
df = buckets_df.merge(
    sp_long[sp_long['node_id']=='TH_SP15_GEN-APND'],
    on=['date','hour'], how='left'
)
df = df.merge(
    load_long[load_long['nodename']=='CA ISO-TAC'],
    on=['date','hour'], how='left'
)
df = df.merge(
    renew_sp,
    on=['date','hour'], how='left'
)

# Fill NaNs
renew_cols = [c for c in df.columns if c.endswith('_mw')]
for c in renew_cols:
    df[c] = df[c].fillna(0)
df['mw_in_bucket'] = df['mw_in_bucket'].fillna(0)
df = df.sort_values(["date","hour","bucket_low"]).reset_index(drop=True)
df["cum_mw_pred"] = df.groupby(["date","hour"])["mw_in_bucket"].cumsum()

# ------------------------------------------------------------
# 8) SAVE MODELING DATAFRAME
# ------------------------------------------------------------
df_sorted = df.sort_values(['date','hour','bucket_low']).reset_index(drop=True)
df_sorted.to_csv(OUTPUT_MODELING_CSV, index=False)
print("Saved modeling dataframe to", OUTPUT_MODELING_CSV)

# Optional pivot per hour
pivot = df_sorted.pivot_table(index=['date','hour','price','nodename','load_mw'],
                              columns='bucket_label', values='mw_in_bucket', aggfunc='sum', fill_value=0).reset_index()
pivot.to_csv("modeling_pivot_per_hour_buckets.csv", index=False)
print("Saved pivoted per-hour bucket CSV: modeling_pivot_per_hour_buckets.csv")

# ============================================================
# 9) MW per bucket + clearing price prediction (CA ISO-TAC)
# ============================================================
df_model = df[df['nodename'] == 'CA ISO-TAC'].copy()
df_model['bucket_mid'] = (df_model['bucket_low'] + df_model['bucket_high']) / 2
df_model_full = df_model.copy()

# Feature engineering
features = ['bucket_low','bucket_high','hour','load_mw','solar_mw','wind_mw']
target = 'mw_in_bucket'
for f in features:
    df_model[f] = pd.to_numeric(df_model[f], errors='coerce').fillna(0)

# Train-test split
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost

parameter = {'colsample_bytree': np.float64(0.8446612641953124), 'gamma': np.float64(0.03533152609858703), 'learning_rate': np.float64(0.01691872751242473), 'max_depth': 13, 'min_child_weight': 3, 'n_estimators': 1131, 'reg_alpha': np.float64(4.667628932479799), 'reg_lambda': np.float64(8.599404067363206), 'subsample': np.float64(0.8721230154351118)}

# XGBoost
model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42, **parameter)
model.fit(X_train, y_train)


fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop features (MW in Bucket Prediction):")
print(fi.head(30))
fi.head(30).plot(kind='barh', figsize=(8,10))
plt.title('Volume model feature importance')
plt.tight_layout()
plt.show()


# Predict & evaluate
y_pred2 = model.predict(X_train)
y_pred = model.predict(X_test)
print("Train RMSE:", mean_squared_error(y_train, y_pred2))
print("Train MAE: ", mean_absolute_error(y_train, y_pred2))
print("Train R2:", r2_score(y_train, y_pred2))
print("Test RMSE:", mean_squared_error(y_test, y_pred))
print("Test MAE: ", mean_absolute_error(y_test, y_pred))
print("Test R2:", r2_score(y_test, y_pred))

# Predict MW for all CA ISO-TAC rows
df_model_full['mw_in_bucket_pred'] = model.predict(df_model_full[features])
df_model_full['cum_mw_pred'] = df_model_full.groupby(['date','hour'])['mw_in_bucket_pred'].cumsum()
df_model_full['cum_mw'] = df_model_full.groupby(['date','hour'])['mw_in_bucket'].cumsum()

plt.figure(figsize=(6,6))
plt.scatter(df_model_full['cum_mw'], df_model_full['cum_mw_pred'], alpha=0.4)
lims = [min(df_model_full['cum_mw'].min(), df_model_full['cum_mw_pred'].min()), max(df_model_full['cum_mw'].max(), df_model_full['cum_mw_pred'].max())]
plt.plot(lims, lims, linestyle='--', color='r', label='y = x')
plt.xlabel('Actual Cumulative Prices')
plt.ylabel('Actual Cumulative Prices')
plt.title('Inital Model - Actual vs Predicted Megawatts per Price Bucket for Cumulative Program')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(df_model_full['mw_in_bucket'], df_model_full['mw_in_bucket_pred'], alpha=0.4)
lims = [min(df_model_full['mw_in_bucket'].min(), df_model_full['mw_in_bucket_pred'].min()), max(df_model_full['mw_in_bucket'].max(), df_model_full['mw_in_bucket_pred'].max())]
plt.plot(lims, lims, linestyle='--', color='r', label='y = x')
plt.xlabel('Actual Megawatts Per Bucket')
plt.ylabel('Predicted Megawatts Per Bucket')
plt.title('Initial Model - Actual vs Predicted Megawatts Per Bucket')
plt.grid(True)
plt.tight_layout()
plt.show()


































# ----------------------------
# Upgraded price modeling: regimes + net-load + multi-lags + tuned XGB
# Requires df_model_full, sp_long, GAS_PRICES_CSV (optional)
# ----------------------------
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

# --- 0) Parameters / regimes ---
REGIME_BINS = [-1e9, 150, 400, 1e9]   # [normal (<150), high (150-400), extreme (>400)]
REGIME_NAMES = ['normal', 'high', 'extreme']
LAGS_DAYS = [1, 2, 3] 
TIME_TEST_FRACTION = 0.20  # last 20% of time for test
RANDOM_STATE = 42

# --- 1) Build hourly aggregated dataset (features) ---
# reuse earlier hourly creation (hourly from previous block). If not, build here:
# hourly: must include date, hour, load_mw, solar_mw, wind_mw, total_supply_pred, pred_clear_price_inv, cum_at_* thresholds, etc.
# If your script already produced `hourly` and `hourly_train`, use them; otherwise reconstruct here quickly:
# We'll reconstruct from df_model_full + sp_long to be safe.

# 1a: base hourly from df_model_full
hourly_base = (df_model_full
               .sort_values(['date','hour','bucket_low'])
               .groupby(['date','hour'])
               .agg({
                   'load_mw':'first',
                   'solar_mw':'first',
                   'wind_mw':'first',
                   'price':'first'   # this is SP15 price from the bucket join (actual price)
               }).reset_index().rename(columns={'price':'price_raw'}))

# 1b: totals & thresholds (reuse your thresholds variable if present)
try:
    thresholds
except NameError:
    thresholds = [-25, 0, 5, 10, 15, 20, 50, 100, 200]

hour_total_supply = df_model_full.groupby(['date','hour'])['cum_mw_pred'].max().reset_index().rename(columns={'cum_mw_pred':'total_supply_pred'})

def cum_at_price_for_group(g, thresh):
    row = g[g['bucket_high'] >= thresh]
    if row.empty:
        return float(g['cum_mw_pred'].max())
    return float(row['cum_mw_pred'].iloc[0])

hourly_thresh_rows = []
for (d,h), g in df_model_full.groupby(['date','hour']):
    rec = {'date':d, 'hour':h}
    for t in thresholds:
        rec[f'cum_at_{t}'] = cum_at_price_for_group(g, t)
    hourly_thresh_rows.append(rec)
hourly_thresh = pd.DataFrame(hourly_thresh_rows)

# 1c: predicted inverse clearing price feature
def predicted_clearing_price_row(g):
    load = g['load_mw'].iloc[0]
    total = g['cum_mw_pred'].max()
    if load >= total:
        return float(g['bucket_high'].max())
    row = g[g['cum_mw_pred'] >= load]
    return float(row['bucket_high'].iloc[0]) if not row.empty else float(g['bucket_high'].max())

pred_inv_rows = []
for (d,h), g in df_model_full.groupby(['date','hour']):
    pred_inv_rows.append({'date':d, 'hour':h, 'pred_clear_price_inv': predicted_clearing_price_row(g)})
pred_inv = pd.DataFrame(pred_inv_rows)

# 1d: merge to hourly
hourly = hourly_base.merge(hour_total_supply, on=['date','hour'], how='left')
hourly = hourly.merge(hourly_thresh, on=['date','hour'], how='left')
hourly = hourly.merge(pred_inv, on=['date','hour'], how='left')

# 1e: actual SP15 price label (prefer sp_long price if available)
sp = sp_long.copy()
sp['date'] = pd.to_datetime(sp['date'])
hourly['date'] = pd.to_datetime(hourly['date'])
label = sp[['date','hour','price']].rename(columns={'price':'price_actual'})
hourly = hourly.merge(label, on=['date','hour'], how='left')

# drop rows where label missing (can't train on those)
hourly = hourly.dropna(subset=['price_actual']).reset_index(drop=True)

# --- 2) Add net-load + ramps (B) ---
hourly['net_load'] = hourly['load_mw'] - hourly['solar_mw'].fillna(0) - hourly['wind_mw'].fillna(0)
hourly = hourly.sort_values(['date','hour']).reset_index(drop=True)
hourly['net_load_ramp'] = hourly['net_load'].diff().fillna(0)
hourly['load_ramp'] = hourly['load_mw'].diff().fillna(0)
hourly['renewables'] = hourly['solar_mw'].fillna(0) + hourly['wind_mw'].fillna(0)
hourly['renewables_ramp'] = hourly['renewables'].diff().fillna(0)

# --- 3) Add lag features (C) ---
# We create lags for price_actual (these are available historically; they won't be at prediction time,
# but they are useful features for backtesting and for models that will be run online with lag inputs)
lag_features_daily = []

for L in LAGS_DAYS:
    for f in ['price_actual', 'pred_clear_price_inv', 'total_supply_pred', 'load_mw']:
        col_name = f'{f}_lag{L}d'
        hourly[col_name] = hourly.groupby('hour')[f].shift(L).fillna(method='ffill').fillna(0)
        lag_features_daily.append(col_name)
        
# --- 4) Merge gas prices (optional) ---
def load_gas_prices(gcsv):
    g = pd.read_csv(gcsv, low_memory=False)
    g.columns = [c.strip() for c in g.columns]
    day_cols = [c for c in g.columns if re.fullmatch(r'd0?\d', c)]
    if not day_cols:
        day_cols = [c for c in g.columns if re.match(r'(?i)^d', c)]
    g_long = g.melt(id_vars=[c for c in g.columns if c not in day_cols], value_vars=day_cols,
                    var_name='d', value_name='gas_price')
    g_long['day'] = g_long['d'].str.extract(r'(\d+)').astype(int)
    g_long['opr_month'] = pd.to_datetime(g_long['opr_month'])
    g_long['date'] = g_long['opr_month'] + pd.to_timedelta(g_long['day']-1, unit='D')
    gp = g_long.groupby(['date'])['gas_price'].mean().reset_index()
    return gp

try:
    gp = load_gas_prices(GAS_PRICES_CSV)
    hourly = hourly.merge(gp, on='date', how='left')
    hourly['gas_price'] = pd.to_numeric(hourly['gas_price'], errors='coerce').fillna(method='ffill').fillna(0)
except Exception:
    hourly['gas_price'] = 0.0

# --- 5) Regime label by observed price (A) ---
hourly['regime'] = pd.cut(hourly['price_actual'], bins=REGIME_BINS, labels=REGIME_NAMES)
# quick counts
print("Regime counts:\n", hourly['regime'].value_counts())

# --- 6) Final feature list (careful, avoid massive col explosion) ---
base_features = [
    'load_mw','solar_mw','wind_mw','net_load','net_load_ramp','load_ramp','renewables','renewables_ramp',
    'total_supply_pred','pred_clear_price_inv', 'gas_price','hour']
# add threshold cum_at_* features
cum_features = [c for c in hourly.columns if c.startswith('cum_at_')]

feature_cols = base_features + cum_features + lag_features_daily

# keep only columns that exist
feature_cols = [c for c in feature_cols if c in hourly.columns]
print("Number of features used:", len(feature_cols))

# --- 7) Time-based train/test split (strict) ---
hourly = hourly.sort_values(['date','hour']).reset_index(drop=True)
# split by index so we don't leak future info; keep last TIME_TEST_FRACTION for test
split_idx = int(len(hourly) * (1.0 - TIME_TEST_FRACTION))
train = hourly.iloc[:split_idx].copy()
test = hourly.iloc[split_idx:].copy()

print("Train rows:", len(train), "Test rows:", len(test))

# --- 8) Helper: sample weights to upweight rare extremes ---
# Weight >1 for extreme regime so model pays attention to rare spikes
def compute_sample_weights(df):
    # base weight = 1. increase for extreme regime
    w = np.ones(len(df), dtype=float)
    w[df['regime']=='high'] *= 2.0
    w[df['regime']=='extreme'] *= 5.0
    return w

from sklearn.ensemble import RandomForestRegressor

rf_parameters =  {'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 8, 'max_features': 0.6, 'max_depth': 12}

# --- 9) Train a global fallback model + one per regime (A+D) ---

model_global = RandomForestRegressor(random_state=42, **rf_parameters)

# Train global model (fallback)
print("Training global Random Forest model...")
w_train = compute_sample_weights(train)
model_global.fit(train[feature_cols], train['price_actual'],
                 sample_weight=w_train)


# --- 10) Evaluate models: global and regime-specific on test set ---
def eval_preds(y_true, y_pred, prefix=""):
    rmse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix} RMSE: {rmse:.3f}  MAE: {mae:.3f}  R2: {r2:.3f}")
    return rmse, mae, r2

print("\nGlobal model evaluation on test set:")

y_pred_global2 = model_global.predict(train[feature_cols])
y_pred_global = model_global.predict(test[feature_cols])
eval_preds(train['price_actual'], y_pred_global2, prefix="Train_Global")
eval_preds(test['price_actual'], y_pred_global, prefix="Global")


# Save test predictions (choose global + regime fallback)
test_out = test.copy()
test_out['pred_global'] = y_pred_global

test_out['pred_regime'] = test_out['pred_global'].copy()
test_out[['date','hour','price_actual','pred_global','pred_regime']].to_csv("price_test_predictions_regimes.csv", index=False)

# --- 11) Feature importance (global) ---
# Create dataframe of feature importances
fi_df = (
    pd.DataFrame({
        'feature': feature_cols,
        'importance': model_global.feature_importances_
    })
    .sort_values('importance', ascending=False)
    .reset_index(drop=True)
)

print(fi_df.head(30))

# --- 12) Residual diagnostics (global) ---
res = test['price_actual'] - test_out['pred_regime']
plt.figure(figsize=(6,4))
plt.hist(res, bins=100)
plt.axvline(0, color='k', linestyle='--')
plt.title('Residuals (actual - pred_regime)')
plt.tight_layout()
plt.show()


# Create a DataFrame of actual vs predicted
df_pred_actual = test[['date','hour','price_actual']].copy()
df_pred_actual['predicted_price'] = y_pred_global

print('This is how the actual test compares to test predicted:')
print(df_pred_actual.head())





plt.figure(figsize=(6,6))
plt.scatter(test['price_actual'], y_pred_global, alpha=0.4)
lims = [min(test['price_actual'].min(), y_pred_global.min()), max(test['price_actual'].max(), y_pred_global.max())]
plt.plot(lims, lims, linestyle='--', color='r', label='y = x')
plt.xlabel('Actual SP15 Price')
plt.ylabel('Predicted Clearing Price')
plt.title('Final Model Actual vs Predicted Clearing Price - Test Data')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()





import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try cvxpy for convex constraints
use_cvxpy = True
try:
    import cvxpy as cp
except:
    use_cvxpy = False
    from sklearn.linear_model import Ridge

# -----------------------
# BUILD Q_obs FROM BUCKETS
# -----------------------

# buckets in ascending order
bucket_edges = [-25, 0, 5, 10, 15, 20, 50, 100, 200]
bucket_cols  = ['cum_at_-25','cum_at_0','cum_at_5','cum_at_10','cum_at_15',
                'cum_at_20','cum_at_50','cum_at_100','cum_at_200']

def compute_Qobs(row):
    load = row['load_mw']
    # find first bucket that exceeds load
    for col in bucket_cols:
        if row[col] >= load:
            return row[col]
    return row['cum_at_200']


hourly['Q_obs'] = hourly.apply(compute_Qobs, axis=1)
hourly['Qk'] = hourly['Q_obs'] / 1000.0   # scale for stability

# -----------------------
# SELECT FEATURES
# -----------------------

exog_cols = ['net_load', 'renewables', 'gas_price']
regime_col = 'regime'
use_regime = True

y = hourly['price_actual'].values
Qk = hourly['Qk'].values

# Build exogenous matrix: [1, net_load, renewables, gas_price]
X_exog = np.column_stack([np.ones(len(hourly))] + [hourly[c].values for c in exog_cols])
exog_names = ['const'] + exog_cols

# -----------------------
# REGIME HANDLING (beta_r)
# -----------------------

if use_regime:
    regimes = sorted(hourly[regime_col].astype(str).unique())
    R_mat = np.zeros((len(hourly), len(regimes)))
    for i, r in enumerate(regimes):
        R_mat[:, i] = (hourly[regime_col].astype(str) == r).astype(float)
    regime_names = regimes
else:
    R_mat = np.ones((len(hourly), 1))
    regime_names = ['global']

# -----------------------
# SOLVE CONSTRAINED PROBLEM
# -----------------------

if use_cvxpy:
    theta = cp.Variable(X_exog.shape[1])      # intercept + exog
    beta  = cp.Variable(R_mat.shape[1], nonneg=True)  # beta_r >= 0 (convex slope)

    beta_effect = cp.multiply(R_mat @ beta, Qk)
    y_pred = X_exog @ theta + beta_effect

    lam = 1e-3
    obj = cp.sum_squares(y - y_pred) + lam*(cp.sum_squares(theta) + cp.sum_squares(beta))
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=cp.OSQP, verbose=False)

    theta_hat = theta.value
    beta_hat = beta.value
    hourly['price_pred_inv'] = (X_exog @ theta_hat) + (R_mat @ beta_hat)*Qk

else:
    # Fallback (no constraints)
    from sklearn.linear_model import Ridge
    X_comb = np.hstack([X_exog, R_mat * Qk.reshape(-1,1)])
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_comb, y)
    hourly['price_pred_inv'] = ridge.predict(X_comb)

    theta_hat = ridge.coef_[:len(exog_names)]
    beta_hat = ridge.coef_[len(exog_names):]

# -----------------------
# METRICS
# -----------------------

rmse = np.sqrt(mean_squared_error(y, hourly['price_pred_inv']))
mae  = mean_absolute_error(y, hourly['price_pred_inv'])
r2   = r2_score(y, hourly['price_pred_inv'])

print("Inverse Optimization Fit Results")
print(f"Rows: {len(hourly)}")
print(f"RMSE = {rmse:.3f}, MAE = {mae:.3f}, R2 = {r2:.3f}\n")

print("Exogenous Coefficients:")
for name, val in zip(exog_names, theta_hat):
    print(f"  {name:12s} = {val:.6f}")

print("\nConvex slope coefficients (beta per regime):")
for rn, bv in zip(regime_names, beta_hat):
    print(f"  beta[{rn:8s}] = {bv:.6f}")

# Save output
hourly[['date','hour','price_actual','price_pred_inv','Q_obs']].to_csv("inverseopt_hourly_results.csv", index=False)
print("\nSaved inverseopt_hourly_results.csv")

import pandas as pd
import matplotlib.pyplot as plt

# Load the results (or use your existing dataframe)
df = pd.read_csv("inverseopt_hourly_results.csv")
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = df['date'] + pd.to_timedelta(df['hour']-1, unit='h')

# -------------------------
# 1) Time series plot: actual vs predicted
# -------------------------
plt.figure(figsize=(14,5))
plt.plot(df['datetime'], df['price_actual'], label='Actual Price', alpha=0.8)
plt.plot(df['datetime'], df['price_pred_inv'], label='Predicted Price (Inverse Opt)', alpha=0.8)
plt.xlabel("Datetime")
plt.ylabel("SP15 Price ($/MWh)")
plt.title("Actual vs Inverse-Optimization Predicted Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 2) Parity plot: predicted vs actual
# -------------------------
plt.figure(figsize=(6,6))
plt.scatter(df['price_actual'], df['price_pred_inv'], alpha=0.3)
lims = [df[['price_actual','price_pred_inv']].min().min(),
        df[['price_actual','price_pred_inv']].max().max()]
plt.plot(lims, lims, 'r--', lw=1.5)
plt.xlabel("Actual Price ($/MWh)")
plt.ylabel("Predicted Price ($/MWh)")
plt.title("Parity Plot: Predicted vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 3) Residual histogram (optional)
# -------------------------
residuals = df['price_actual'] - df['price_pred_inv']
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=80, alpha=0.7)
plt.axvline(0, color='k', linestyle='--')
plt.title("Residuals (Actual - Predicted)")
plt.xlabel("Price Residual ($/MWh)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
