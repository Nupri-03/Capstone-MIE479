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
from sklearn.metrics import mean_squared_error, r2_score
# ---------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------- USER PATHS / SETTINGS ----------
BIDS_ZIP_OR_FOLDER = r"\\VSRV2\C.Homes$\karathah\Desktop\Data Required for CAPSTONE Modelling\2024 DAM Bid Data.zip"
LOAD_CSV =  r"\\VSRV2\C.Homes$\karathah\Desktop\Data Required for CAPSTONE Modelling\load_forecasts_2024.csv"
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
model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred2 = model.predict(X_train)
y_pred = model.predict(X_test)
print("Train RMSE:", mean_squared_error(y_train, y_pred2))
print("Train R2:", r2_score(y_train, y_pred2))
print("Test RMSE:", mean_squared_error(y_test, y_pred))
print("Test R2:", r2_score(y_test, y_pred))

# Predict MW for all CA ISO-TAC rows
df_model_full['mw_in_bucket_pred'] = model.predict(df_model_full[features])
df_model_full['cum_mw_pred'] = df_model_full.groupby(['date','hour'])['mw_in_bucket_pred'].cumsum()
df_model_full['cum_mw'] = df_model_full.groupby(['date','hour'])['mw_in_bucket'].cumsum()


plt.figure(figsize=(6,6))
plt.scatter(df_model_full['cum_mw'], df_model_full['cum_mw_pred'], alpha=0.4)
plt.xlabel('Actual SP15 Price')
plt.ylabel('Predicted Clearing Price')
plt.title('Inital Model - Actual vs Predicted Clearing Price for Cumulative Program')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(df_model_full['mw_in_bucket'], df_model_full['mw_in_bucket_pred'], alpha=0.4)
plt.xlabel('Actual SP15 Price')
plt.ylabel('Predicted Clearing Price')
plt.title('Initial Model - Actual vs Predicted Clearing Price Per Bucket')
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
LAGS = [1, 2, 24]
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
for L in LAGS:
    hourly[f'price_actual_lag{L}'] = hourly['price_actual'].shift(L).fillna(method='ffill').fillna(0)
    hourly[f'pred_clear_price_inv_lag{L}'] = hourly['pred_clear_price_inv'].shift(L).fillna(method='ffill').fillna(0)
    hourly[f'total_supply_pred_lag{L}'] = hourly['total_supply_pred'].shift(L).fillna(method='ffill').fillna(0)
    hourly[f'load_mw_lag{L}'] = hourly['load_mw'].shift(L).fillna(method='ffill').fillna(0)

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
lag_features = []
for L in LAGS:
    lag_features += [f'price_actual_lag{L}', f'pred_clear_price_inv_lag{L}', f'total_supply_pred_lag{L}', f'load_mw_lag{L}']

feature_cols = base_features + cum_features + lag_features

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

# --- 9) Train a global fallback model + one per regime (A+D) ---
xgb_common_params = dict(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    tree_method='hist'  
)

# Train global model (fallback)
print("Training global XGBoost model...")
model_global = XGBRegressor(**xgb_common_params)
w_train = compute_sample_weights(train)
model_global.fit(train[feature_cols], train['price_actual'],
                 sample_weight=w_train,
                 eval_set=[(train[feature_cols], train['price_actual']), (test[feature_cols], test['price_actual'])],
                 verbose=50)


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
fi = pd.Series(model_global.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop features (global):")
print(fi.head(30))
fi.head(30).plot(kind='barh', figsize=(8,10))
plt.title('Global model feature importance')
plt.tight_layout()
plt.show()

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





'''

# =========================
# Predict SP15 price using full feature set (XGBoost)
# Requires: df_model_full (hour-bucket rows), sp_long (hourly actual prices),
# and GAS_PRICES_CSV path if you want gas features.
# =========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ---------- 0) Constants ----------
# if you used BUCKET_EDGES in your script, reuse it; else define sensible thresholds
try:
    BUCKET_EDGES  # noqa
except NameError:
    BUCKET_EDGES = list(range(-25, 205, 5))

# ---------- 1) Aggregate df_model_full to hourly features ----------
# df_model_full should contain one row per bucket per (date,hour)
# We'll build features by pivoting / aggregating across buckets.

# 1.a: hourly base features (load, renewables) - keep first since they are identical per hour
hourly_base = (df_model_full
               .sort_values(['date','hour','bucket_low'])
               .groupby(['date','hour'])
               .agg({
                   'load_mw':'first',
                   'solar_mw':'first',
                   'wind_mw':'first'
               }).reset_index())

# 1.b: total predicted supply at top-of-stack
hour_total_supply = df_model_full.groupby(['date','hour'])['cum_mw_pred'].max().reset_index().rename(columns={'cum_mw_pred':'total_supply_pred'})

# 1.c: supply at selected price thresholds (cum MW at price p)
# choose thresholds to cover relevant range: e.g., -25, 0, 5, 10, 20, 50, 100
thresholds = [-25, 0, 5, 10, 15, 20, 50, 100, 200]
def cum_at_price_for_group(g, thresh):
    # g sorted by bucket_high ascending (should be)
    # find first bucket_high >= thresh and return cum_mw_pred at that row
    row = g[g['bucket_high'] >= thresh]
    if row.empty:
        # thresh beyond top bucket; return total
        return g['cum_mw_pred'].max()
    return float(row['cum_mw_pred'].iloc[0])

hourly_thresh = []
for (d,h), g in df_model_full.groupby(['date','hour']):
    rec = {'date':d, 'hour':h}
    for t in thresholds:
        rec[f'cum_at_{t}'] = cum_at_price_for_group(g, t)
    hourly_thresh.append(rec)
hourly_thresh = pd.DataFrame(hourly_thresh)

# 1.d: bucket statistics over predicted buckets: moments of the predicted distribution
agg_stats = (df_model_full
             .groupby(['date','hour'])
             .agg({
                 'mw_in_bucket_pred':['sum','mean','median','std'],
                 'bucket_mid': 'first' if 'bucket_mid' in df_model_full.columns else ('bucket_low','first')
             }))
# flatten multiindex
agg_stats.columns = ['_'.join(map(str,c)).strip() for c in agg_stats.columns]
agg_stats = agg_stats.reset_index()

# 1.e: merge hourly features
hourly = hourly_base.merge(hour_total_supply, on=['date','hour'], how='left')
hourly = hourly.merge(hourly_thresh, on=['date','hour'], how='left')
# merge agg stats if present
hourly = hourly.merge(agg_stats, on=['date','hour'], how='left')

# ---------- 2) Add predicted clearing price from inversion as a feature ----------
# you already have code for that; compute a simple predicted clearing price feature:
def predicted_clearing_price_row(g):
    load = g['load_mw'].iloc[0]
    total = g['cum_mw_pred'].max()
    if load >= total:
        # price at max bucket_high
        return float(g['bucket_high'].max())
    row = g[g['cum_mw_pred'] >= load]
    return float(row['bucket_high'].iloc[0]) if not row.empty else float(g['bucket_high'].max())

pred_clears = []
for (d,h), g in df_model_full.groupby(['date','hour']):
    pred_clears.append({'date':d, 'hour':h, 'pred_clear_price_inv': predicted_clearing_price_row(g)})
pred_clears = pd.DataFrame(pred_clears)
hourly = hourly.merge(pred_clears, on=['date','hour'], how='left')

# ---------- 3) Merge actual SP15 price (label) ----------
# sp_long must have columns date, hour, price, node_id
sp = sp_long.copy()
# If node_id uses TH_SP15_GEN-APND, filter earlier; but sp_long may already be SP15
# Ensure date/hour types match
sp['date'] = pd.to_datetime(sp['date'])
hourly['date'] = pd.to_datetime(hourly['date'])
label = sp[['date','hour','price']].rename(columns={'price':'price_actual'})
hourly = hourly.merge(label, on=['date','hour'], how='left')

# Drop hours with missing actual price (can't train on those)
hourly_train = hourly.dropna(subset=['price_actual']).copy()
print("Hourly rows with label:", len(hourly_train))

# ---------- 4) Merge gas prices (optional) ----------
# GAS_PRICES_CSV format (as you showed) has monthly rows with d01..d31
def load_gas_prices(gcsv):
    g = pd.read_csv(gcsv, low_memory=False)
    g.columns = [c.strip() for c in g.columns]
    # find day columns like d01..d31
    day_cols = [c for c in g.columns if re.fullmatch(r'd0?\d', c)]
    if not day_cols:
        # some files use d01..d31 or 1..31; try pattern
        day_cols = [c for c in g.columns if re.match(r'(?i)^d', c)]
    # melt to day
    g_long = g.melt(id_vars=[c for c in g.columns if c not in day_cols], value_vars=day_cols,
                    var_name='d', value_name='gas_price')
    # extract day number
    g_long['day'] = g_long['d'].str.extract(r'(\d+)').astype(int)
    # construct date from opr_month (first day) + day-1
    g_long['opr_month'] = pd.to_datetime(g_long['opr_month'])
    g_long['date'] = g_long['opr_month'] + pd.to_timedelta(g_long['day']-1, unit='D')
    # pick a representative gas price (if multiple nodes exist select mean)
    gp = g_long.groupby(['date'])['gas_price'].mean().reset_index()
    return gp

try:
    gp = load_gas_prices(GAS_PRICES_CSV)
    hourly_train = hourly_train.merge(gp, on='date', how='left')
    hourly_train['gas_price'] = pd.to_numeric(hourly_train['gas_price'], errors='coerce').fillna(method='ffill').fillna(0)
except Exception as e:
    print("Gas load failed (optional). Continuing without gas feature:", e)
    hourly_train['gas_price'] = 0.0

# ---------- 5) Add time & lag features ----------
hourly_train['hour_of_day'] = hourly_train['hour'].astype(int)
hourly_train['dow'] = pd.to_datetime(hourly_train['date']).dt.dayofweek
hourly_train['month'] = pd.to_datetime(hourly_train['date']).dt.month

# sort and build lags
hourly_train = hourly_train.sort_values(['date','hour']).reset_index(drop=True)
# build lagged actual price, lagged predicted clearing price, lagged load
hourly_train['price_actual_lag1'] = hourly_train['price_actual'].shift(1)
hourly_train['pred_clear_price_inv_lag1'] = hourly_train['pred_clear_price_inv'].shift(1)
hourly_train['total_supply_pred_lag1'] = hourly_train['total_supply_pred'].shift(1)
hourly_train['load_mw_lag1'] = hourly_train['load_mw'].shift(1)
# For beginning rows where lag is NaN, fill with zeros or forward-fill
hourly_train[['price_actual_lag1','pred_clear_price_inv_lag1','total_supply_pred_lag1','load_mw_lag1']] = \
    hourly_train[['price_actual_lag1','pred_clear_price_inv_lag1','total_supply_pred_lag1','load_mw_lag1']].fillna(method='ffill').fillna(0)

# ---------- 6) Final features and train/test split ----------
feature_cols = [
    'load_mw','solar_mw','wind_mw',
    'total_supply_pred','pred_clear_price_inv',
    'price_actual_lag1','pred_clear_price_inv_lag1','total_supply_pred_lag1','load_mw_lag1',
    'gas_price','hour_of_day','dow','month'] + [f'cum_at_{t}' for t in thresholds]

# ensure all feature cols exist
feature_cols = [c for c in feature_cols if c in hourly_train.columns]

X = hourly_train[feature_cols]
y = hourly_train['price_actual']

# train/test split by time (preserve temporal order) or random
# here we'll do a time-based split: last 20% as test
hourly_train = hourly_train.sort_values(["date", "hour"])
split_idx = int(len(hourly_train)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ---------- 7) Train XGBoost ----------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)

# ---------- 8) Evaluate ----------

y_pred2 = model.predict(X_train)
rmse = mean_squared_error(y_train, y_pred2)
mae = mean_absolute_error(y_train, y_pred2)
r2 = r2_score(y_train, y_pred2)
print("Train RMSE:", rmse)
print("Train MAE :", mae)
print("Train R2  :", r2)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Test RMSE:", rmse)
print("Test MAE :", mae)
print("Test R2  :", r2)

# Save predictions
pred_df = X_test.copy()
pred_df['price_actual'] = y_test
pred_df['price_pred'] = y_pred

# ---------- 9) Feature importance ----------
fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("Feature importance (top 20):")
print(fi.head(20))
fi.head(20).plot(kind='barh', figsize=(8,6))
plt.title('Feature importance')
plt.tight_layout()
plt.savefig("feature_importance_price_model.png")
plt.show()

# ---------- 10) Diagnostics: scatter & residuals ----------
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.4)
mn = min(y_test.min(), y_pred.min()); mx = max(y_test.max(), y_pred.max())
plt.plot([mn,mx],[mn,mx],'--', color='k')
plt.xlabel('Actual price'); plt.ylabel('Predicted price'); plt.title('Actual vs Predicted SP15')
plt.tight_layout(); plt.savefig("price_scatter_xy.png"); plt.show()

plt.figure(figsize=(6,4))
res = y_test - y_pred
plt.hist(res, bins=80)
plt.axvline(0, linestyle='--')
plt.title('Residuals (actual - pred)')
plt.tight_layout()
plt.show()







'''
























'''

STOP HERE

def predicted_clearing_price(df):
    # df is one hour of data, with predicted cum_mw_pred and bucket_high
    
    # If load > total supply, clear at max bucket
    total_supply = df['cum_mw_pred'].max()
    load = df['load_mw'].iloc[0]

    if load >= total_supply:
        return df['bucket_high'].max()  # price at max supply

    # otherwise find the lowest bucket meeting the load
    row = df[df['cum_mw_pred'] >= load].iloc[0]
    return row['bucket_high']

predicted_prices = []

for (d, h), dfh in df_model_full.groupby(['date','hour']):
    p = predicted_clearing_price(dfh)
    predicted_prices.append({
        'date': d,
        'hour': h,
        'predicted_price': p,
        'actual_price': dfh['price'].iloc[0]
    })

pred_prices_df = pd.DataFrame(predicted_prices)


df_price_compare = df_model_full.merge(pred_prices_df, on=['date','hour'], how='left')




# ---------------------------
# Pricing accuracy & diagnostics
# If you already have clearing_compare with columns:
# 'date','hour','clearing_price_pred','clearing_price_actual'
# use it; otherwise, compute it (fallback)
try:
    cc = clearing_compare.copy()
except NameError:
    # recompute predicted clearing price quickly (uses df_model_full)
    def predicted_clearing_price_rows(dfh):
        total_supply = dfh['cum_mw_pred'].max()
        load = dfh['load_mw'].iloc[0]
        if load >= total_supply:
            return dfh['bucket_high'].max()
        row = dfh[dfh['cum_mw_pred'] >= load]
        if row.empty:
            return dfh['bucket_high'].max()
        return float(row['bucket_high'].iloc[0])

    preds = []
    for (d,h), g in df_model_full.groupby(['date','hour']):
        preds.append({
            'date': d,
            'hour': h,
            'clearing_price_pred': predicted_clearing_price_rows(g),
            'clearing_price_actual': g['price'].iloc[0]  # actual price column
        })
    cc = pd.DataFrame(preds)

# drop rows where actual is missing
cc = cc.dropna(subset=['clearing_price_actual']).copy()

# Convert to numeric
cc['clearing_price_pred'] = pd.to_numeric(cc['clearing_price_pred'], errors='coerce')
cc['clearing_price_actual'] = pd.to_numeric(cc['clearing_price_actual'], errors='coerce')
cc = cc.dropna(subset=['clearing_price_pred','clearing_price_actual']).reset_index(drop=True)

# Error terms
cc['err'] = cc['clearing_price_pred'] - cc['clearing_price_actual']
cc['abs_err'] = cc['err'].abs()
# MAPE: avoid division by zero by excluding (or use alternative)
mask_nonzero = cc['clearing_price_actual'] != 0
cc['ape'] = np.nan
cc.loc[mask_nonzero, 'ape'] = (cc.loc[mask_nonzero, 'abs_err'] / cc.loc[mask_nonzero, 'clearing_price_actual']).abs()

# Metrics
rmse = mean_squared_error(cc['clearing_price_actual'], cc['clearing_price_pred'])
mae = mean_absolute_error(cc['clearing_price_actual'], cc['clearing_price_pred'])
r2 = r2_score(cc['clearing_price_actual'], cc['clearing_price_pred'])
mape = cc['ape'].mean() * 100  # percent
bias = cc['err'].mean()
medae = cc['abs_err'].median()

# Hit rates
for tol in [1,5,10]:
    hits = (cc['abs_err'] <= tol).sum()
    total = len(cc)
    print(f"Pct hours |err| ≤ ${tol:2d}: {hits}/{total} = {100*hits/total:.2f}%")

print("\nSummary metrics:")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"Median AE: {medae:.3f}")
print(f"Bias (mean error): {bias:.3f}")
print(f"R^2 : {r2:.4f}")
print(f"MAPE: {mape:.2f}% (only computed where actual!=0)")

# Per-hour RMSE (to see which hours are hardest)
hourly_rmse = cc.groupby('hour').apply(
    lambda g: mean_squared_error(g['clearing_price_actual'], g['clearing_price_pred'])
).rename('rmse').reset_index()
print("\nHourly RMSE (sample):")
print(hourly_rmse.head(24).to_string(index=False))

# Per-month RMSE
cc['month'] = pd.to_datetime(cc['date']).dt.to_period('M')
monthly_rmse = cc.groupby('month').apply(
    lambda g: mean_squared_error(g['clearing_price_actual'], g['clearing_price_pred'])
).rename('rmse').reset_index()
print("\nMonthly RMSE (sample):")
print(monthly_rmse.head().to_string(index=False))

# Save metrics and diagnostics
metrics = {
    'rmse': rmse, 'mae': mae, 'medae': medae, 'bias': bias, 'r2': r2, 'mape_pct': mape,
    'n_obs': len(cc)
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("pricing_accuracy_metrics.csv", index=False)
hourly_rmse.to_csv("pricing_hourly_rmse.csv", index=False)
monthly_rmse.to_csv("pricing_monthly_rmse.csv", index=False)
cc.to_csv("clearing_price_by_hour_with_errors.csv", index=False)
print("\nSaved metrics and per-hour/month CSVs.")

# ---------------------------
# Plots: Actual vs Predicted and residual histogram
# ---------------------------
# Scatter: actual vs predicted
plt.figure(figsize=(6,6))
plt.scatter(cc['clearing_price_actual'], cc['clearing_price_pred'], alpha=0.4)
mn = min(cc['clearing_price_actual'].min(), cc['clearing_price_pred'].min())
mx = max(cc['clearing_price_actual'].max(), cc['clearing_price_pred'].max())
plt.plot([mn,mx], [mn,mx], linestyle='--')
plt.xlabel('Actual SP15 Price')
plt.ylabel('Predicted Clearing Price')
plt.title('Actual vs Predicted Clearing Price')
plt.grid(True)
plt.tight_layout()
plt.savefig("price_scatter_actual_vs_predicted.png")
plt.show()

# Residual histogram
plt.figure(figsize=(6,4))
plt.hist(cc['err'], bins=80)
plt.axvline(0, linestyle='--')
plt.xlabel('Prediction Error (pred - actual)')
plt.ylabel('Count')
plt.title('Residual Histogram')
plt.tight_layout()
plt.savefig("price_residual_histogram.png")
plt.show()


'''
'''

# Compute predicted clearing price
def get_clearing_price(group):
    load = group['load_mw'].iloc[0]
    idx = np.searchsorted(group['cum_mw_pred'].values, load)
    if idx == 0:
        return group['price'].iloc[0]
    elif idx >= len(group):
        return group['price'].iloc[-1]
    else:
        mw_lo = group['cum_mw_pred'].iloc[idx-1]
        mw_hi = group['cum_mw_pred'].iloc[idx]
        price_lo = group['price'].iloc[idx-1]
        price_hi = group['price'].iloc[idx]
        if mw_hi == mw_lo:
            return price_hi
        return price_lo + (load - mw_lo)/(mw_hi - mw_lo)*(price_hi - price_lo)

clearing_prices = df_model_full.groupby(['date','hour']).apply(get_clearing_price).reset_index()
clearing_prices.columns = ['date','hour','clearing_price_pred']

# Merge actual SP15 prices
sp15_prices = sp_long[sp_long['node_id'] == 'TH_SP15_GEN-APND']
sp15_prices = sp15_prices[['date','hour','price']].rename(columns={'price':'clearing_price_actual'})

clearing_compare = clearing_prices.merge(sp15_prices, on=['date','hour'], how='left')

# Evaluate
mask = ~clearing_compare['clearing_price_actual'].isna()
rmse_price = mean_squared_error(clearing_compare.loc[mask,'clearing_price_actual'],
                                clearing_compare.loc[mask,'clearing_price_pred'])
r2_price = r2_score(clearing_compare.loc[mask,'clearing_price_actual'],
                    clearing_compare.loc[mask,'clearing_price_pred'])
print("Clearing price RMSE:", rmse_price)
print("Clearing price R2:", r2_price)

# Save predictions
df_model_full.to_csv("df_model_mw_predictions.csv", index=False)
clearing_compare.to_csv("clearing_price_predictions.csv", index=False)
print("Saved MW predictions and clearing price predictions.")

'''
