#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:52:10 2025

@author: juliarice
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

TARGET_ASSET = 1000776

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ---------------------------
# User config
# ---------------------------

BASE = "/Users/juliarice/Desktop/capstone/"
ASSET_XREF_FILE = os.path.join(BASE, "asset_xref_202509121109.csv")
BID_FOLDER = os.path.join(BASE, "2024 DAM Bid Data")
GAS_PRICE_FILE = os.path.join(BASE, "Energy_Gas_Prices_2024.csv")
LOAD_FCST_FILE = os.path.join(BASE, "load_tesla_fcst_202511162319.csv")
OUT_DIR = os.path.join(BASE, f"{TARGET_ASSET}rf_model_output")
os.makedirs(OUT_DIR, exist_ok=True)

LOOKBACK_DAYS = 365
SPLIT_DATE = pd.to_datetime("2024-11-01")  # Train before, val after

MODEL_TYPE = 'rf'

GBM_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "subsample": 0.8,
    "random_state": 42
}

RF_PARAMS = {
    "n_estimators": 150,
    "random_state": 42,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "n_jobs": -1
}

CLUSTER_SUBSAMPLE = 3000
TRAIN_SUBSAMPLE_PER_CLUSTER = 5000
MIN_K = 3
MAX_K = 10
FIXED_K = None

# ---------------------------
# Helpers
# ---------------------------
def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return pd.DataFrame()

def load_bid_data(folder_path, x_days=365):
    if not os.path.isdir(folder_path):
        print(f"Warning: bid folder {folder_path} not found")
        return pd.DataFrame()
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    files = files[:min(len(files), x_days)]
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"Warning reading {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ---------------------------
# Load datasets
# ---------------------------
print("Loading data...")
asset_xref = safe_read_csv(ASSET_XREF_FILE)
bids_all = load_bid_data(BID_FOLDER, LOOKBACK_DAYS)
gas = safe_read_csv(GAS_PRICE_FILE)
load_fcst = safe_read_csv(LOAD_FCST_FILE)

# ---------------------------
# Merge and filter
# ---------------------------
if not asset_xref.empty and "asset_xref" in asset_xref.columns:
    merged = pd.merge(asset_xref, bids_all, on="asset_xref", how="inner")
else:
    merged = bids_all.copy()

if "resource_type" in merged.columns:
    merged = merged[merged["resource_type"].astype(str).str.upper() == "GENERATOR"].copy()

bids = merged[merged["asset_xref"] == TARGET_ASSET].copy()
if bids.empty:
    raise RuntimeError(f"No bid rows found for asset_xref={TARGET_ASSET}")

# ---------------------------
# Date/time normalization
# ---------------------------
if "opr_date" in bids.columns:
    bids["opr_date"] = pd.to_datetime(bids["opr_date"], errors="coerce")
elif "dyn_datetime_int" in bids.columns:
    bids["dyn_datetime_int"] = pd.to_numeric(bids["dyn_datetime_int"], errors="coerce")
    bids["opr_date"] = (pd.to_datetime("1900-01-01") + pd.to_timedelta(bids["dyn_datetime_int"], "h")).dt.floor("D")
else:
    bids["opr_date"] = pd.NaT

if "hour_ending" in bids.columns:
    bids["hour_ending"] = pd.to_numeric(bids["hour_ending"], errors="coerce").round(0).astype("Int64")
    bids.loc[bids["hour_ending"] == 0, "hour_ending"] = 24
else:
    if "dyn_datetime_int" in bids.columns:
        tmp_dt = pd.to_datetime(bids["dyn_datetime_int"], unit="h", origin=pd.Timestamp("1900-01-01"))
        bids["hour_ending"] = (tmp_dt.dt.hour + 1).astype("Int64")
    else:
        bids["hour_ending"] = pd.NA

bids["quantity"] = pd.to_numeric(bids.get("quantity", 0), errors="coerce").fillna(0)
bids["price1"] = pd.to_numeric(bids.get("price1", 0), errors="coerce").fillna(0)

# Filter economic bids
econ_bids = bids[bids["price1"] < 900].copy()
if econ_bids.empty:
    raise RuntimeError("No economic bids found")

# ---------------------------
# Merge external data
# ---------------------------
load_long = pd.DataFrame()
if not load_fcst.empty:
    lf = load_fcst.copy()
    lf["opr_date"] = pd.to_datetime(lf.get("opr_date", None), errors="coerce")
    hour_cols = [c for c in lf.columns if re.match(r'he\d+', c.lower())]
    if hour_cols:
        load_long = lf.melt(id_vars=["opr_date"], value_vars=hour_cols,
                            var_name="hour_col", value_name="load_mw")
        load_long["hour_ending"] = load_long["hour_col"].str.extract(r'(\d+)').astype(int)
        load_long = load_long[["opr_date", "hour_ending", "load_mw"]]

gas_long = pd.DataFrame()
if not gas.empty:
    g = gas.copy()
    if "opr_month" in g.columns:
        try:
            g["opr_month"] = pd.to_datetime(g["opr_month"], errors="coerce")
            day_cols = [c for c in g.columns if c.lower().startswith("d")]
            if day_cols:
                gas_long = g.melt(id_vars=["opr_month"], value_vars=day_cols,
                                  var_name="day_col", value_name="gas_price")
                gas_long["day"] = gas_long["day_col"].str.extract(r'(\d+)').astype(int)
                gas_long["opr_date"] = gas_long["opr_month"] + pd.to_timedelta(gas_long["day"] - 1, unit="D")
                gas_long = gas_long[["opr_date", "gas_price"]]
        except Exception:
            gas_long = pd.DataFrame()

# ---------------------------
# Build base dataframe
# ---------------------------
df = econ_bids.copy()
if not load_long.empty:
    df = df.merge(load_long, on=["opr_date", "hour_ending"], how="left")
if not gas_long.empty:
    df = df.merge(gas_long, on=["opr_date"], how="left")

df = df.dropna(subset=["load_mw", "price1", "opr_date", "hour_ending"])
df = df.sort_values(["opr_date", "hour_ending"]).reset_index(drop=True)

print(f"Total rows after merge: {len(df)}")

# ---------------------------
# SPLIT TRAIN/VAL FIRST (before any feature engineering)
# ---------------------------
train = df[df["opr_date"] < SPLIT_DATE].copy().reset_index(drop=True)
val = df[df["opr_date"] >= SPLIT_DATE].copy().reset_index(drop=True)

print(f"Train: {len(train)} rows ({train['opr_date'].min()} to {train['opr_date'].max()})")
print(f"Val: {len(val)} rows ({val['opr_date'].min()} to {val['opr_date'].max()})")

# ---------------------------
# FEATURE ENGINEERING (separately for train and val, no leakage)
# ---------------------------
def engineer_features(data, is_train=True, train_stats=None):
    """
    Engineer features without target leakage.
    - All lag/rolling features use .shift() to exclude current row
    - No price1 used in features
    - Rolling stats computed only from past
    """
    df_feat = data.copy()
    
    # Basic temporal
    df_feat["month"] = df_feat["opr_date"].dt.month
    df_feat["dow"] = df_feat["opr_date"].dt.dayofweek
    df_feat["day"] = df_feat["opr_date"].dt.day
    df_feat["is_weekend"] = (df_feat["dow"] >= 5).astype(int)
    df_feat["season"] = df_feat["month"] % 12 // 3 + 1
    df_feat["is_summer"] = ((df_feat["month"] >= 6) & (df_feat["month"] <= 9)).astype(int)
    
    # Peak indicators
    df_feat["is_peak"] = ((df_feat["hour_ending"] >= 7) & (df_feat["hour_ending"] <= 22)).astype(int)
    df_feat["is_super_peak"] = ((df_feat["hour_ending"] >= 16) & (df_feat["hour_ending"] <= 21)).astype(int)
    
    # Cyclical hour encoding
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour_ending"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour_ending"] / 24)
    
    # Gas price handling
    if "gas_price" in df_feat.columns:
        if is_train:
            gas_median = df_feat["gas_price"].median()
        else:
            gas_median = train_stats['gas_median']
        df_feat["gas_price"] = df_feat["gas_price"].fillna(gas_median)
        
        # FIXED: Shifted gas rolling mean (no leakage)
        df_feat["gas_roll7_mean"] = df_feat["gas_price"].shift(1).rolling(window=7, min_periods=3).mean()
        df_feat["gas_roll7_mean"] = df_feat["gas_roll7_mean"].fillna(df_feat["gas_price"])
    else:
        gas_median = 0.0
        df_feat["gas_price"] = 0.0
        df_feat["gas_roll7_mean"] = 0.0
    
    # FIXED: Lag features (previous day same hour) - properly shifted
    # Group by hour and shift to get previous occurrence
    df_feat = df_feat.sort_values(["hour_ending", "opr_date"]).reset_index(drop=True)
    df_feat["price_lag1"] = df_feat.groupby("hour_ending")["price1"].shift(1)
    df_feat["qty_lag1"] = df_feat.groupby("hour_ending")["quantity"].shift(1)
    df_feat["load_lag1"] = df_feat.groupby("hour_ending")["load_mw"].shift(1)
    
    # FIXED: Rolling statistics - shifted to exclude current value
    df_feat["price_roll7_mean"] = df_feat.groupby("hour_ending")["price1"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=3).mean()
    )
    df_feat["price_roll7_std"] = df_feat.groupby("hour_ending")["price1"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=3).std()
    )
    
    # Re-sort by date
    df_feat = df_feat.sort_values(["opr_date", "hour_ending"]).reset_index(drop=True)
    
    # Load normalization (using training stats)
    if is_train:
        load_stats = df_feat.groupby("month")["load_mw"].agg(['mean', 'std']).to_dict()
    else:
        load_stats = train_stats['load_stats']
    
    df_feat["load_normalized"] = df_feat.apply(
        lambda row: (row["load_mw"] - load_stats['mean'].get(row["month"], row["load_mw"])) / 
                    (load_stats['std'].get(row["month"], 1) + 1e-6),
        axis=1
    )
    
    # REMOVED: heat_rate_proxy (was using price1 - target leakage)
    
    stats_dict = {
        'gas_median': gas_median,
        'load_stats': load_stats
    } if is_train else None
    
    return df_feat, stats_dict

print("\nEngineering features (train)...")
train, train_stats = engineer_features(train, is_train=True)

print("Engineering features (val)...")
val, _ = engineer_features(val, is_train=False, train_stats=train_stats)

# ---------------------------
# CLUSTERING (fit on training data only, NO price1)
# ---------------------------
print("\nPerforming clustering on TRAINING data only...")

# FIXED: Use quantity + hour features (NO price1), proper cyclical encoding
cluster_features = ["quantity", "hour_sin", "hour_cos"]
cluster_src_train = train[cluster_features].copy().dropna()

if len(cluster_src_train) > CLUSTER_SUBSAMPLE:
    cluster_for_fit = cluster_src_train.sample(CLUSTER_SUBSAMPLE, random_state=42)
else:
    cluster_for_fit = cluster_src_train

scaler_cluster = StandardScaler()
X_cluster_fit = scaler_cluster.fit_transform(cluster_for_fit)

# Find best k
if FIXED_K is not None:
    best_k = FIXED_K
    print(f"Using fixed k={best_k}")
else:
    best_k = 4
    best_score = -1.0
    for k in range(MIN_K, min(MAX_K, len(cluster_for_fit) // 10) + 1):
        try:
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_tmp = km_tmp.fit_predict(X_cluster_fit)
            score = silhouette_score(X_cluster_fit, labels_tmp)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    print(f"Selected k={best_k} (silhouette={best_score:.3f})")

# Fit KMeans on training data
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_cluster_fit)

# FIXED: Predict clusters with proper alignment
# For train
train_cluster_data = train[cluster_features].copy()
train_cluster_scaled = scaler_cluster.transform(train_cluster_data.fillna(0))
train["cluster"] = kmeans.predict(train_cluster_scaled)

# For val
val_cluster_data = val[cluster_features].copy()
val_cluster_scaled = scaler_cluster.transform(val_cluster_data.fillna(0))
val["cluster"] = kmeans.predict(val_cluster_scaled)

# Cluster analysis
print("\nCluster Analysis (from training data):")
for c in sorted(train["cluster"].unique()):
    subset = train[train["cluster"] == c]
    print(f"Cluster {c}: n={len(subset)}, "
          f"avg_price=${subset['price1'].mean():.1f}, "
          f"avg_qty={subset['quantity'].mean():.1f}MW, "
          f"peak_hrs={subset['is_peak'].mean()*100:.0f}%")

# ---------------------------
# Define features (NO LEAKAGE)
# ---------------------------
base_features = ["load_mw", "gas_price", "month", "dow", "hour_ending",
                 "is_weekend", "is_peak", "is_super_peak", "is_summer",
                 "hour_sin", "hour_cos", "load_normalized", "gas_roll7_mean"]

# Lag features (all properly shifted)
lag_features = ["price_lag1", "qty_lag1", "load_lag1", 
                "price_roll7_mean", "price_roll7_std"]

features = base_features + lag_features

# Fill remaining NaNs with training medians
feature_medians = {}
for feat in features:
    if feat in train.columns:
        feature_medians[feat] = train[feat].median()
        train[feat] = train[feat].fillna(feature_medians[feat])
        val[feat] = val[feat].fillna(feature_medians[feat])

train = train.dropna(subset=features + ["price1", "quantity"]).reset_index(drop=True)
val = val.dropna(subset=features + ["price1", "quantity"]).reset_index(drop=True)

print(f"\nAfter feature engineering: Train={len(train)}, Val={len(val)}")

# ---------------------------
# Model Training
# ---------------------------
print(f"\nTraining {MODEL_TYPE.upper()} models per cluster...")

models_price = {}
models_qty = {}
cluster_results = []

for c in sorted(train["cluster"].unique()):
    tr = train[train["cluster"] == c]
    va = val[val["cluster"] == c]
    
    if len(tr) < 30 or len(va) < 10:
        print(f"Skipping cluster {c} (insufficient data: train={len(tr)}, val={len(va)})")
        continue
    
    if len(tr) > TRAIN_SUBSAMPLE_PER_CLUSTER:
        tr = tr.sample(TRAIN_SUBSAMPLE_PER_CLUSTER, random_state=42)
    
    X_tr = tr[features]
    y_tr_price = tr["price1"]
    y_tr_qty = tr["quantity"]
    
    X_va = va[features]
    y_va_price = va["price1"]
    y_va_qty = va["quantity"]
    
    # Price model
    if MODEL_TYPE == 'gbm':
        m_price = GradientBoostingRegressor(**GBM_PARAMS)
    else:
        m_price = RandomForestRegressor(**RF_PARAMS)
    
    m_price.fit(X_tr, y_tr_price)
    pred_price = m_price.predict(X_va)
    mae_price = mean_absolute_error(y_va_price, pred_price)
    r2_price = r2_score(y_va_price, pred_price)
    
    # Quantity model
    if MODEL_TYPE == 'gbm':
        m_qty = GradientBoostingRegressor(**GBM_PARAMS)
    else:
        m_qty = RandomForestRegressor(**RF_PARAMS)
    
    m_qty.fit(X_tr, y_tr_qty)
    pred_qty = m_qty.predict(X_va)
    mae_qty = mean_absolute_error(y_va_qty, pred_qty)
    r2_qty = r2_score(y_va_qty, pred_qty)
    
    models_price[c] = m_price
    models_qty[c] = m_qty
    cluster_results.append({
        'cluster': c,
        'price_MAE': mae_price,
        'price_R2': r2_price,
        'qty_MAE': mae_qty,
        'qty_R2': r2_qty,
        'train_size': len(tr),
        'val_size': len(va)
    })
    
    print(f"Cluster {c}: Price MAE=${mae_price:.2f} R²={r2_price:.3f}, "
          f"Qty MAE={mae_qty:.1f}MW R²={r2_qty:.3f}")

# ---------------------------
# Check for overfitting
# ---------------------------
print("\n" + "="*60)
print("OVERFITTING CHECK: Train vs Validation Performance")
print("="*60)

train_pred_price = []
train_actual_price = []
train_pred_qty = []
train_actual_qty = []

for c, group in train.groupby("cluster"):
    if c in models_price:
        Xg = group[features]
        train_pred_price.extend(models_price[c].predict(Xg))
        train_actual_price.extend(group["price1"].values)
        train_pred_qty.extend(models_qty[c].predict(Xg))
        train_actual_qty.extend(group["quantity"].values)

train_metrics = {
    'price_R2': r2_score(train_actual_price, train_pred_price),
    'price_MAE': mean_absolute_error(train_actual_price, train_pred_price),
    'qty_R2': r2_score(train_actual_qty, train_pred_qty),
    'qty_MAE': mean_absolute_error(train_actual_qty, train_pred_qty)
}

# ---------------------------
# Validation predictions
# ---------------------------
val_out = val.copy()
val_out["pred_price"] = np.nan
val_out["pred_qty"] = np.nan

for c, group in val_out.groupby("cluster"):
    if c in models_price:
        Xg = group[features]
        val_out.loc[group.index, "pred_price"] = models_price[c].predict(Xg)
        val_out.loc[group.index, "pred_qty"] = models_qty[c].predict(Xg)

val_out["price_error"] = val_out["pred_price"] - val_out["price1"]
val_out["qty_error"] = val_out["pred_qty"] - val_out["quantity"]
val_out["price_pct_error"] = (val_out["price_error"] / val_out["price1"]) * 100

# ---------------------------
# Metrics
# ---------------------------
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

val_metrics = {
    "price_MAE": mean_absolute_error(val_out["price1"], val_out["pred_price"]),
    "price_RMSE": np.sqrt(mean_squared_error(val_out["price1"], val_out["pred_price"])),
    "price_MAPE": mape(val_out["price1"], val_out["pred_price"]),
    "price_R2": r2_score(val_out["price1"], val_out["pred_price"]),
    "qty_MAE": mean_absolute_error(val_out["quantity"], val_out["pred_qty"]),
    "qty_RMSE": np.sqrt(mean_squared_error(val_out["quantity"], val_out["pred_qty"])),
    "qty_MAPE": mape(val_out["quantity"], val_out["pred_qty"]),
    "qty_R2": r2_score(val_out["quantity"], val_out["pred_qty"])
}

# Print comparison

print(f"  Train R²: {train_metrics['price_R2']:.4f}, MAE: ${train_metrics['price_MAE']:.4f}")
print(f"  Val   R²: {val_metrics['price_R2']:.4f}, MAE: ${val_metrics['price_MAE']:.4f}")
print(f"  Gap:      {train_metrics['price_R2'] - val_metrics['price_R2']:.4f}, ${train_metrics['price_MAE'] - val_metrics['price_MAE']:.4f}")


print(f"  Train R²: {train_metrics['qty_R2']:.4f}, MAE: {train_metrics['qty_MAE']:.4f}MW")
print(f"  Val   R²: {val_metrics['qty_R2']:.4f}, MAE: {val_metrics['qty_MAE']:.4f}MW")
print(f"  Gap:      {train_metrics['qty_R2'] - val_metrics['qty_R2']:.4f}, {train_metrics['qty_MAE'] - val_metrics['qty_MAE']:.4f}MW")

print("\n" + "="*60)
print("OVERALL VALIDATION METRICS")
print("="*60)
for k, v in val_metrics.items():
    print(f"{k:20s}: {v:10.4f}")

# Per-hour metrics
hour_metrics = val_out.groupby("hour_ending").apply(
    lambda g: pd.Series({
        'price_MAE': mean_absolute_error(g["price1"], g["pred_price"]),
        'qty_MAE': mean_absolute_error(g["quantity"], g["pred_qty"]),
        'actual_price': g["price1"].mean(),
        'pred_price': g["pred_price"].mean(),
        'actual_qty': g["quantity"].mean(),
        'pred_qty': g["pred_qty"].mean()
    })
).round(2)

# ---------------------------
# PLOTTING (same as before, just using corrected data)
# ---------------------------
print("\nGenerating plots...")

# 1. Price Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].scatter(val_out["price1"], val_out["pred_price"], alpha=0.4, s=10)
axes[0, 0].plot([val_out["price1"].min(), val_out["price1"].max()],
                [val_out["price1"].min(), val_out["price1"].max()], 'r--', lw=2)
axes[0, 0].set_xlabel("Actual Price ($/MWh)", fontsize=12)
axes[0, 0].set_ylabel("Predicted Price ($/MWh)", fontsize=12)
axes[0, 0].set_title(f"Price Prediction (R²={val_metrics['price_R2']:.3f})", fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(val_out["price_error"], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='red', linestyle='--', lw=2)
axes[0, 1].set_xlabel("Price Error ($/MWh)", fontsize=12)
axes[0, 1].set_ylabel("Frequency", fontsize=12)
axes[0, 1].set_title(f"Price Error Distribution (MAE=${val_metrics['price_MAE']:.2f})", 
                     fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

sample_days = val_out[val_out["opr_date"].isin(val_out["opr_date"].unique()[:7])]
axes[1, 0].plot(range(len(sample_days)), sample_days["price1"], 'o-', label="Actual", alpha=0.7)
axes[1, 0].plot(range(len(sample_days)), sample_days["pred_price"], 's-', label="Predicted", alpha=0.7)
axes[1, 0].set_xlabel("Time Steps (7 days sample)", fontsize=12)
axes[1, 0].set_ylabel("Price ($/MWh)", fontsize=12)
axes[1, 0].set_title("Price Time Series (Sample)", fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].bar(hour_metrics.index, hour_metrics["price_MAE"], edgecolor='black', alpha=0.7)
axes[1, 1].axhline(val_metrics['price_MAE'], color='red', linestyle='--', lw=2, 
                   label=f'Overall MAE: ${val_metrics["price_MAE"]:.2f}')
axes[1, 1].set_xlabel("Hour Ending", fontsize=12)
axes[1, 1].set_ylabel("MAE ($/MWh)", fontsize=12)
axes[1, 1].set_title("Price MAE by Hour", fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "price_analysis.png"), dpi=300, bbox_inches='tight')

plt.close()

# 2. Quantity Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].scatter(val_out["quantity"], val_out["pred_qty"], alpha=0.4, s=10)
axes[0, 0].plot([val_out["quantity"].min(), val_out["quantity"].max()],
                [val_out["quantity"].min(), val_out["quantity"].max()], 'r--', lw=2)
axes[0, 0].set_xlabel("Actual Quantity (MW)", fontsize=12)
axes[0, 0].set_ylabel("Predicted Quantity (MW)", fontsize=12)
axes[0, 0].set_title(f"Quantity Prediction (R²={val_metrics['qty_R2']:.3f})", fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(val_out["qty_error"], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='red', linestyle='--', lw=2)
axes[0, 1].set_xlabel("Quantity Error (MW)", fontsize=12)
axes[0, 1].set_ylabel("Frequency", fontsize=12)
axes[0, 1].set_title(f"Quantity Error Distribution (MAE={val_metrics['qty_MAE']:.1f}MW)", 
                     fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

sample_days = val_out[val_out["opr_date"].isin(val_out["opr_date"].unique()[:7])]
axes[1, 0].plot(range(len(sample_days)), sample_days["quantity"], 'o-', label="Actual", alpha=0.7)
axes[1, 0].plot(range(len(sample_days)), sample_days["pred_qty"], 's-', label="Predicted", alpha=0.7)
axes[1, 0].set_xlabel("Time Steps (7 days sample)", fontsize=12)
axes[1, 0].set_ylabel("Quantity (MW)", fontsize=12)
axes[1, 0].set_title("Quantity Time Series (Sample)", fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].bar(hour_metrics.index, hour_metrics["qty_MAE"], edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].axhline(val_metrics['qty_MAE'], color='red', linestyle='--', lw=2, 
                   label=f'Overall MAE: {val_metrics["qty_MAE"]:.1f}MW')
axes[1, 1].set_xlabel("Hour Ending", fontsize=12)
axes[1, 1].set_ylabel("MAE (MW)", fontsize=12)
axes[1, 1].set_title("Quantity MAE by Hour", fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "quantity_analysis.png"), dpi=300, bbox_inches='tight')

plt.close()

# 3. Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

importances_price = []
for c, model in models_price.items():
    if hasattr(model, 'feature_importances_'):
        importances_price.append(model.feature_importances_)

if importances_price:
    avg_importance_price = np.mean(importances_price, axis=0)
    imp_df_price = pd.DataFrame({
        'feature': features,
        'importance': avg_importance_price
    }).sort_values('importance', ascending=False).head(15)
    
    axes[0].barh(range(len(imp_df_price)), imp_df_price['importance'], edgecolor='black')
    axes[0].set_yticks(range(len(imp_df_price)))
    axes[0].set_yticklabels(imp_df_price['feature'])
    axes[0].set_xlabel("Importance", fontsize=12)
    axes[0].set_title("Top 15 Features: Price Model", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

importances_qty = []
for c, model in models_qty.items():
    if hasattr(model, 'feature_importances_'):
        importances_qty.append(model.feature_importances_)

if importances_qty:
    avg_importance_qty = np.mean(importances_qty, axis=0)
    imp_df_qty = pd.DataFrame({
        'feature': features,
        'importance': avg_importance_qty
    }).sort_values('importance', ascending=False).head(15)
    
    axes[1].barh(range(len(imp_df_qty)), imp_df_qty['importance'], edgecolor='black', color='orange')
    axes[1].set_yticks(range(len(imp_df_qty)))
    axes[1].set_yticklabels(imp_df_qty['feature'])
    axes[1].set_xlabel("Importance", fontsize=12)
    axes[1].set_title("Top 15 Features: Quantity Model", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Cluster Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

cluster_counts_train = train['cluster'].value_counts().sort_index()
axes[0, 0].bar(cluster_counts_train.index, cluster_counts_train.values, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel("Cluster", fontsize=12)
axes[0, 0].set_ylabel("Count (Training)", fontsize=12)
axes[0, 0].set_title("Cluster Distribution", fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

cluster_stats = train.groupby('cluster').agg({
    'price1': 'mean',
    'quantity': 'mean'
}).reset_index()

axes[0, 1].scatter(cluster_stats['quantity'], cluster_stats['price1'], 
                   s=cluster_counts_train.values/50, alpha=0.6, c=cluster_stats['cluster'], cmap='tab10')
for idx, row in cluster_stats.iterrows():
    axes[0, 1].annotate(f"C{int(row['cluster'])}", 
                        (row['quantity'], row['price1']),
                        fontsize=10, fontweight='bold')
axes[0, 1].set_xlabel("Avg Quantity (MW)", fontsize=12)
axes[0, 1].set_ylabel("Avg Price ($/MWh)", fontsize=12)
axes[0, 1].set_title("Cluster Characteristics", fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

cluster_res_df = pd.DataFrame(cluster_results)
if not cluster_res_df.empty:
    x_pos = np.arange(len(cluster_res_df))
    width = 0.35
    
    axes[1, 0].bar(x_pos - width/2, cluster_res_df['price_MAE'], width, 
                   label='Price MAE', alpha=0.7, edgecolor='black')
    axes[1, 0].bar(x_pos + width/2, cluster_res_df['qty_MAE']/10, width,
                   label='Qty MAE/10', alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel("Cluster", fontsize=12)
    axes[1, 0].set_ylabel("MAE", fontsize=12)
    axes[1, 0].set_title("Model Performance by Cluster", fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f"C{int(c)}" for c in cluster_res_df['cluster']])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].bar(x_pos - width/2, cluster_res_df['price_R2'], width,
                   label='Price R²', alpha=0.7, edgecolor='black')
    axes[1, 1].bar(x_pos + width/2, cluster_res_df['qty_R2'], width,
                   label='Qty R²', alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].set_xlabel("Cluster", fontsize=12)
    axes[1, 1].set_ylabel("R² Score", fontsize=12)
    axes[1, 1].set_title("R² Score by Cluster", fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f"C{int(c)}" for c in cluster_res_df['cluster']])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cluster_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5. Temporal Patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

hourly_avg = val_out.groupby('hour_ending').agg({
    'price1': 'mean',
    'pred_price': 'mean'
})
axes[0, 0].plot(hourly_avg.index, hourly_avg['price1'], 'o-', label='Actual', linewidth=2)
axes[0, 0].plot(hourly_avg.index, hourly_avg['pred_price'], 's-', label='Predicted', linewidth=2)
axes[0, 0].set_xlabel("Hour Ending", fontsize=12)
axes[0, 0].set_ylabel("Average Price ($/MWh)", fontsize=12)
axes[0, 0].set_title("Average Price by Hour", fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

hourly_qty = val_out.groupby('hour_ending').agg({
    'quantity': 'mean',
    'pred_qty': 'mean'
})
axes[0, 1].plot(hourly_qty.index, hourly_qty['quantity'], 'o-', label='Actual', linewidth=2, color='orange')
axes[0, 1].plot(hourly_qty.index, hourly_qty['pred_qty'], 's-', label='Predicted', linewidth=2, color='red')
axes[0, 1].set_xlabel("Hour Ending", fontsize=12)
axes[0, 1].set_ylabel("Average Quantity (MW)", fontsize=12)
axes[0, 1].set_title("Average Quantity by Hour", fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

dow_error = val_out.groupby('dow').agg({
    'price_error': lambda x: np.abs(x).mean(),
    'qty_error': lambda x: np.abs(x).mean()
})
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1, 0].bar(range(7), dow_error['price_error'], edgecolor='black', alpha=0.7)
axes[1, 0].set_xticks(range(7))
axes[1, 0].set_xticklabels(dow_names)
axes[1, 0].set_xlabel("Day of Week", fontsize=12)
axes[1, 0].set_ylabel("Mean Absolute Error ($/MWh)", fontsize=12)
axes[1, 0].set_title("Price MAE by Day of Week", fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

peak_comp = val_out.groupby('is_peak').agg({
    'price_error': lambda x: np.abs(x).mean()
})
x_labels = ['Off-Peak', 'Peak']
x_pos = np.arange(len(x_labels))
axes[1, 1].bar(x_pos, peak_comp['price_error'], edgecolor='black', alpha=0.7, color='green')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(x_labels)
axes[1, 1].set_xlabel("Period", fontsize=12)
axes[1, 1].set_ylabel("Price MAE ($/MWh)", fontsize=12)
axes[1, 1].set_title("Price Accuracy: Peak vs Off-Peak", fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "temporal_patterns.png"), dpi=300, bbox_inches='tight')
plt.close()

# 6. Residual Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].scatter(val_out['pred_price'], val_out['price_error'], alpha=0.3, s=10)
axes[0, 0].axhline(0, color='red', linestyle='--', lw=2)
axes[0, 0].set_xlabel("Predicted Price ($/MWh)", fontsize=12)
axes[0, 0].set_ylabel("Price Residual ($/MWh)", fontsize=12)
axes[0, 0].set_title("Price Residuals vs Predicted", fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(val_out['pred_qty'], val_out['qty_error'], alpha=0.3, s=10, color='orange')
axes[0, 1].axhline(0, color='red', linestyle='--', lw=2)
axes[0, 1].set_xlabel("Predicted Quantity (MW)", fontsize=12)
axes[0, 1].set_ylabel("Quantity Residual (MW)", fontsize=12)
axes[0, 1].set_title("Quantity Residuals vs Predicted", fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

stats.probplot(val_out['price_error'].dropna(), dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot: Price Residuals", fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

correlation = val_out[['price_error', 'qty_error']].corr()
im = axes[1, 1].imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(['Price Error', 'Qty Error'])
axes[1, 1].set_yticklabels(['Price Error', 'Qty Error'])
axes[1, 1].set_title("Error Correlation", fontsize=14, fontweight='bold')
for i in range(2):
    for j in range(2):
        text = axes[1, 1].text(j, i, f'{correlation.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=14)
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residual_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# 7. Overfitting Diagnostic Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Price metrics comparison
price_metrics = ['R²', 'MAE']
price_train = [train_metrics['price_R2'], train_metrics['price_MAE']]
price_val = [val_metrics['price_R2'], val_metrics['price_MAE']]

x = np.arange(len(price_metrics))
width = 0.35

axes[0].bar(x - width/2, price_train, width, label='Train', alpha=0.8)
axes[0].bar(x + width/2, price_val, width, label='Val', alpha=0.8)
axes[0].set_ylabel('Value', fontsize=12)
axes[0].set_title('Price Metrics: Train vs Validation', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(price_metrics)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Quantity metrics comparison
qty_metrics = ['R²', 'MAE']
qty_train = [train_metrics['qty_R2'], train_metrics['qty_MAE']]
qty_val = [val_metrics['qty_R2'], val_metrics['qty_MAE']]

axes[1].bar(x - width/2, qty_train, width, label='Train', alpha=0.8)
axes[1].bar(x + width/2, qty_val, width, label='Val', alpha=0.8)
axes[1].set_ylabel('Value', fontsize=12)
axes[1].set_title('Quantity Metrics: Train vs Validation', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(qty_metrics)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "overfitting_diagnostic.png"), dpi=300, bbox_inches='tight')

plt.close()