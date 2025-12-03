import pandas as pd

correct_answer = pd.read_excel(r"C:\Users\harvi\Desktop\Capstone Info\GEN_META_CAPSTONE.xlsx")

jacob_matched = pd.read_excel(r"C:\Users\harvi\Desktop\Capstone Info\jacob-matched.xlsx")

battery_confirmed = pd.read_excel(r"C:\Users\harvi\Desktop\Capstone Info\battery_code_output.xlsx")

gas_confirmed = pd.read_excel(r"C:\Users\harvi\Desktop\Capstone Info\gas_code_output.xlsx")

hydro_confirmed = pd.read_excel(r"C:\Users\harvi\Desktop\Capstone Info\hydro_code_output.xlsx")

hydro_confirmed['is_hydro'] = 'True'

solar_confirmed = pd.read_excel(r"C:\Users\harvi\Desktop\Capstone Info\solar_code_output.xlsx")

wind_confirmed = pd.read_excel(r"C:\Users\harvi\Desktop\Capstone Info\wind_code_output.xlsx")

combo = pd.merge(gas_confirmed, battery_confirmed, on='asset_xref')
combo = combo[['asset_xref','is_thermal_socal', 'is_thermal_pge', 'is_battery_candidate']]

combo = pd.merge(combo, hydro_confirmed, on='asset_xref', how='left')

combo = pd.merge(combo, solar_confirmed, on='asset_xref', how='left')

combo = pd.merge(combo, wind_confirmed, on='asset_xref', how='left')

combo['is_gas'] = combo['is_thermal_socal'] + combo['is_thermal_pge']

cols = ['is_hydro', 'is_battery_candidate','is_hydro', 'is_solar_candidate', 'is_likely_wind', 'is_gas']

combo['true_count'] = combo[cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).sum(axis=1)

final_output = combo[['is_hydro', 'asset_xref', 'is_thermal_socal', 'is_thermal_pge', 'is_battery_candidate', 'is_solar_candidate', 'is_likely_wind', 'is_gas', 'true_count']]

counts = final_output['true_count'].value_counts().sort_index()


checking = pd.merge(final_output, jacob_matched, on='asset_xref', how='left')

mapping = {
    "GAS": "is_gas", 
    "PEAKER": "is_gas", 
    "COMBINED GAS CYCLE": "is_gas", 
    "COGEN": "is_gas",
    "BATTERY": "is_battery_candidate",
    "SOLAR": "is_solar_candidate",
    "WIND": "is_likely_wind",
    "HYDRO": "is_hydro",
}

import numpy as np

def meta_matches(row):
    meta = row['Generalized Meta']
    if pd.isna(meta):
        return np.nan
    if meta not in mapping:
        return False
    col = mapping[meta]
    return row[col] == True

checking["meta_match"] = checking.apply(meta_matches, axis=1)


print("Based on the provided codes that we made for each resource type, here is the accuracy for all the models in total: ")
print(counts)
print("0 means no code picked up this asset")
print("1 means one resource code picked up this asset")
print("2 means two resource codes picked up this asset")


counts2 = checking['meta_match'].value_counts().sort_index()
print(counts2)
print("Out of the 113 assets Jacob found, this is the split on what was accurately picked versus not picked with the created resource codes")

counts2 = counts/(counts[0] +  counts[1] + counts[2])
print(counts2)

