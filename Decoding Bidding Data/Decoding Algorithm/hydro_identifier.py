import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import glob

# ============================================================
# === LOAD DATA ==============================================
# ============================================================
BASE = "Data Required for CAPSTONE Modelling"

ASSET_XREF_FILE     = os.path.join(BASE, "asset_xrefs.csv")
PRICING_FILE        = os.path.join(BASE, "hourly_pricing_data.csv")
BID_FOLDER          = os.path.join(BASE, "2024 DAM Bid Data.zip") 

#asset_type = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\asset_xrefs.csv")
asset_type = pd.read_csv(ASSET_XREF_FILE)
#electricity_pricing = pd.read_csv(r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\hourly_pricing_data.csv")
electricity_pricing = pd.read_csv(PRICING_FILE)

#folder_path = r"C:\Users\jliu\OneDrive - Dynasty Power\Documents\2023 DAM Bid Data"
#files = sorted(glob.glob(os.path.join(folder_path, "*.csv"))) # Load X days of bidding data
folder_path = BID_FOLDER
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

# ============================================================
# === FUNCTIONS ==============================================
# ============================================================

def compute_sustained_gen_hours(asset_day: pd.DataFrame, price_threshold=100):
    """Compute max consecutive hours generating (self-scheduled or price <= threshold)."""
    df = asset_day.copy()
    df["opr_date"] = pd.to_datetime(df["opr_date"])
    df["hour_ending"] = df["hour_ending"].astype(int)
    df["ts"] = df["opr_date"] + pd.to_timedelta(df["hour_ending"] - 1, unit="h")

    df["is_gen_hour"] = (df["selfschedmw"].fillna(0) > 0) | (df["price1"].fillna(1e9) <= price_threshold)
    df = df.sort_values(["asset_xref", "ts"])

    sustained_data = []
    for asset, group in df.groupby("asset_xref"):
        group = group.sort_values("ts").reset_index(drop=True)
        gen_series = group["is_gen_hour"].astype(int)
        streak = 0
        max_streak = 0
        last_ts = None
        for ts, val in zip(group["ts"], gen_series):
            if val:
                if last_ts is not None and (ts - last_ts).total_seconds() == 3600:
                    streak += 1
                else:
                    streak = 1
                last_ts = ts
            else:
                streak = 0
                last_ts = None
            max_streak = max(max_streak, streak)
        sustained_data.append((asset, max_streak))
    return pd.DataFrame(sustained_data, columns=["asset_xref", "max_sustained_hours"])


def compute_seasonal_ratios(asset_day: pd.DataFrame):
    """Compute spring/summer mean generation ratios for each asset, safely handling missing seasons."""
    df = asset_day.copy()
    df["opr_date"] = pd.to_datetime(df["opr_date"])
    df["month"] = df["opr_date"].dt.month
    df["gen_mw"] = df["quantity"].fillna(0) + df["selfschedmw"].fillna(0)

    # Assign each month to a season
    season_map = {
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "summer", 10: "fall", 11: "fall",
        12: "winter", 1: "winter", 2: "spring"
    }
    df["season"] = df["month"].map(season_map)

    # Compute mean generation by season and pivot
    seasonal_means = df.groupby(["asset_xref", "season"])["gen_mw"].mean().unstack(fill_value=0)

    # Ensure all expected seasonal columns exist
    for season in ["spring", "summer", "fall", "winter"]:
        if season not in seasonal_means.columns:
            seasonal_means[season] = 0

    # Avoid division-by-zero
    mean_all = seasonal_means.mean(axis=1).replace(0, np.nan)

    # Compute ratios safely
    seasonal_means["spring_mean_ratio"] = seasonal_means["spring"] / mean_all
    seasonal_means["summer_mean_ratio"] = seasonal_means["summer"] / mean_all

    # Fill any missing values with 0
    seasonal_means = seasonal_means.fillna(0)

    return seasonal_means.reset_index()



def apply_hydro_filters(candidates: pd.DataFrame):
    """
    Hydro heuristics:
    - Remove if max_sustained_hours < 4 (likely peakers)
    - For max_sustained_hours > 24, keep only if (spring_mean_ratio - summer_mean_ratio) > 0.3
    - Keep only if spring_mean_ratio >= 1.2 and summer_mean_ratio <= 1.5
    """
    before = len(candidates)

    # --- Filter 1: Peakers ---
    peakers = candidates[candidates["max_sustained_hours"] < 4]

    # --- Filter 2: Thermal-like units ---
    thermal_like = candidates[
        (candidates["max_sustained_hours"] > 24) &
        ((candidates["spring_mean_ratio"] - candidates["summer_mean_ratio"]) <= 0.3)
    ]

    # --- Filter 3: Seasonal thresholds (your new rule) ---
    seasonal_fail = candidates[
        (candidates["spring_mean_ratio"] < 1.2) |
        (candidates["summer_mean_ratio"] > 1.5)
    ]

    # --- Apply all filters ---
    filtered = candidates[
        (candidates["max_sustained_hours"] >= 4) &
        ~((candidates["max_sustained_hours"] > 24) &
          ((candidates["spring_mean_ratio"] - candidates["summer_mean_ratio"]) <= 0.3)) &
        (candidates["spring_mean_ratio"] >= 1.2) &
        (candidates["summer_mean_ratio"] <= 1.5)
    ].reset_index(drop=True)

    # --- Summary Table ---
    summary = pd.DataFrame({
        "Total Assets": [before],
        "Removed (Peakers)": [len(peakers)],
        "Removed (Thermal-like)": [len(thermal_like)],
        "Removed (Seasonal Thresholds)": [len(seasonal_fail)],
        "Remaining (Likely Hydro)": [len(filtered)]
    })

    return filtered, summary



# ============================================================
# === MAIN EXECUTION =========================================
# ============================================================

# Compute seasonal ratios
seasonal_ratios = compute_seasonal_ratios(asset_day)

# Compute sustained generation hours
sustained_df = compute_sustained_gen_hours(asset_day, price_threshold=100)

# Merge both sets
candidates = pd.merge(seasonal_ratios, sustained_df, on="asset_xref", how="left")

# Apply hydro filtering rules
candidates_filtered, summary = apply_hydro_filters(candidates)

# ============================================================
# === OUTPUTS ================================================
# ============================================================

print("=== Hydro Candidate Summary ===")
print(summary)
print("\n=== Likely Hydro Units ===")
print(candidates_filtered[["asset_xref", "spring_mean_ratio", "summer_mean_ratio", "max_sustained_hours"]])

# ============================================================
# === PLOTS ==================================================
# ============================================================

plt.figure(figsize=(8,6))
plt.scatter(candidates["spring_mean_ratio"], candidates["max_sustained_hours"], alpha=0.4, label="All Assets")
plt.scatter(candidates_filtered["spring_mean_ratio"], candidates_filtered["max_sustained_hours"],
            color="red", label="Likely Hydro", s=80)
plt.axhline(4, color="grey", linestyle="--", alpha=0.6)
plt.axhline(24, color="grey", linestyle="--", alpha=0.6)
plt.xlabel("Spring Mean Ratio")
plt.ylabel("Max Sustained Hours")
plt.title("Hydro Identification by Seasonality and Sustained Generation")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# ============================================================
# === MONTHLY GENERATION PLOTS WITH CAPACITY =================
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Prepare generation data ---
asset_day["opr_date"] = pd.to_datetime(asset_day["opr_date"])
asset_day["gen_mw"] = asset_day["quantity"].fillna(0) + asset_day["selfschedmw"].fillna(0)
asset_day["month"] = asset_day["opr_date"].dt.month
asset_day["year"] = asset_day["opr_date"].dt.year

# --- Compute maximum generation capacity per asset in 2023 ---
capacity_2023 = (
    asset_day[asset_day["year"] == 2023]
    .groupby("asset_xref")["gen_mw"]
    .max()
    .reset_index()
    .rename(columns={"gen_mw": "max_capacity_2023"})
)

# --- Compute monthly average generation ---
monthly_gen = (
    asset_day.groupby(["asset_xref", "month"])["gen_mw"]
    .mean()
    .reset_index()
)

# --- Filter to hydro candidates ---
hydro_assets = candidates_filtered["asset_xref"].unique()
monthly_gen_filtered = monthly_gen[monthly_gen["asset_xref"].isin(hydro_assets)]

# --- Merge capacity info ---
monthly_gen_filtered = monthly_gen_filtered.merge(capacity_2023, on="asset_xref", how="left")

# --- Create plots ---
n_assets = len(hydro_assets)
cols = 3
rows = int(np.ceil(n_assets / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=True, sharey=False)
axes = axes.flatten()

for i, asset in enumerate(hydro_assets):
    ax = axes[i]
    data = monthly_gen_filtered[monthly_gen_filtered["asset_xref"] == asset]
    if data.empty:
        continue

    max_cap = data["max_capacity_2023"].iloc[0]
    ax.plot(data["month"], data["gen_mw"], marker="o", linewidth=2, color="teal")
    ax.set_title(f"Asset {asset} — Cap: {max_cap:.1f} MW", fontsize=10)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Gen (MW)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 13))
    ax.set_xlim(1, 12)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Monthly Average Generation — Likely Hydro Assets", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()

