import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

wake_model = "NOJ"
wind_farms = ["East_Anglia_TWO", "Sofia", "Thor"]
wind_farm_names = ["East Anglia 2", "Sofia", "Thor"]
aep_models = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "BQ", "RQ", "Uniform_CT"]

results_dir = Path("results/optimization")
n_starts = 30

# Normalize with respect to: "360_WD" or "max"
norm = "360_WD"

# --- Build median tables per wind farm ---
tables = {}
for wf in wind_farms:
    rows = []
    h5_path = results_dir / f"windfarm_{wf}.h5"
    with h5py.File(h5_path, "r") as f:
        for aep_model in aep_models:
            aeps, times = [], []
            grp = f[aep_model][wake_model]
            for start_key in grp.keys():
                attrs = grp[start_key].attrs
                aeps.append(attrs["aep_final"])
                times.append(attrs["time"])
            rows.append({
                "AEP_model": aep_model,
                "median_AEP_GWh": np.median(aeps),
                "median_time_s": np.median(times),
            })
    tables[wf] = pd.DataFrame(rows).set_index("AEP_model")

for wf, df in tables.items():
    print(f"\n=== {wf} (wake model: {wake_model}) ===")
    print(df.to_string(float_format="%.3f"))

# --- Normalize results with respect to highest AEP and lowest runtime ---
normalized_tables = {}
for wf, df in tables.items():
    min_aep = df["median_AEP_GWh"].min()
    min_time = df["median_time_s"].min()
    df_norm = df.copy()
    if norm == "360_WD":
        ref_aep = df.loc["360_WD", "median_AEP_GWh"]
        ref_time = df.loc["360_WD", "median_time_s"]
        df_norm["normalized_AEP"] = 100 * (df["median_AEP_GWh"] - ref_aep) / ref_aep
        df_norm["normalized_time"] = 100 * (df["median_time_s"] - ref_time) / ref_time
    else:
        ref_aep = df["median_AEP_GWh"].max()
        ref_time = df["median_time_s"].min()
        df_norm["normalized_AEP"] = 100 * (df["median_AEP_GWh"] - min_aep) / (ref_aep - min_aep)
        df_norm["normalized_time"] = 100 * (df["median_time_s"] - ref_time) / ref_time
    normalized_tables[wf] = df_norm

for wf, df in normalized_tables.items():
    print(f"\n=== {wf} (wake model: {wake_model}) - Normalized ===")
    print(df.to_string(float_format="%.3f"))


# --- Plot heat map for optimized AEP ---
fig, ax = plt.subplots(figsize=(8, 4))
aep_values = np.array([df["normalized_AEP"].values for df in normalized_tables.values()])
im = ax.imshow(
    aep_values,
    cmap="RdYlGn",
    norm=TwoSlopeNorm(vcenter=0, vmin=np.nanmin(aep_values), vmax=np.nanmax(aep_values)),
    aspect="equal",
    alpha=0.8,
    origin="lower",
)
for wf_idx, wf in enumerate(wind_farms):
    for aep_idx, aep_model in enumerate(aep_models):
        value = normalized_tables[wf].loc[aep_model, "normalized_AEP"]
        ax.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black")
plt.colorbar(im, ax=ax)
plt.xticks(range(len(aep_models)), aep_models, rotation=45)
plt.yticks(range(len(wind_farms)), wind_farm_names)
plt.title(f"Normalized AEP - AEP/AEP_max per farm (wake model: {wake_model})")
plt.tight_layout()
plt.show()

# --- Plot heat map for optimization time ---
fig, ax = plt.subplots(figsize=(8, 4))
time_values = np.array([df["normalized_time"].values for df in normalized_tables.values()])
im = ax.imshow(
    time_values,
    cmap="RdYlGn_r",
    norm=TwoSlopeNorm(vcenter=0, vmin=np.nanmin(time_values), vmax=np.nanmax(time_values)),
    aspect="auto",
    alpha=0.8,
)
for wf_idx, wf in enumerate(wind_farms):
    for aep_idx, aep_model in enumerate(aep_models):
        value = normalized_tables[wf].loc[aep_model, "normalized_time"]
        ax.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black")
plt.colorbar(im, ax=ax)
plt.xticks(range(len(aep_models)), aep_models, rotation=45)
plt.yticks(range(len(wind_farms)), wind_farm_names)
plt.title(f"Normalized Optimization Time - Time/Time_min per farm (wake model: {wake_model})")
plt.tight_layout()
plt.show()