import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

wake_model = "TurbOPark"  # "NOJ", "Gaussian", "TurbOPark"
# wind_farms = ["East_Anglia_TWO", "Sofia", "Thor"]
# wind_farm_names = ["East Anglia 2", "Sofia", "Thor"]
aep_models = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "Uniform_CT", "BQ", "RQ", "SGD"]

wind_farms = ["Hornsea_Project_3_HOW03", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea", "Sofia"]
wind_farm_names = ["Hornsea 3 \n (231 WT)", "Thor \n (72 WT)", "Hornsea 2 \n (55 WT)", "Sofia \n (100 WT)"]

wind_farms = ["Hornsea_Project_3_HOW03"]

results_dir = Path("results/optimization/Iteration_3_4WF_20starts")
n_starts = 20

# Normalize with respect to: "360_WD" or "max"
norm = "360_WD"

# --- Build median tables per wind farm ---
tables = {}
for wf in wind_farms:
    rows = []
    h5_path = f"windfarm_{wf}.h5"
    with h5py.File(h5_path, "r") as f:
        for aep_model in aep_models:
            aeps, times = [], []
            for start_id in range(n_starts):
                group_key = f"{aep_model}/{wake_model}/start_{start_id}"
                if group_key not in f:
                    continue  # No more starts
                grp = f[group_key]
                attrs = grp.attrs
                aeps.append(attrs["aep_final"])
                times.append(attrs["time"])

            print("TOtal starts for ", wf, aep_model, np.count_nonzero(~np.isnan(aeps)))
            rows.append({
                "AEP_model": aep_model,
                "median_AEP_GWh": np.median(aeps),
                "median_time_s": np.median(times),
            })
    tables[wf] = pd.DataFrame(rows).set_index("AEP_model")

# for wf, df in tables.items():
#     print(f"\n=== {wf} (wake model: {wake_model}) ===")
#     print(df.to_string(float_format="%.3f"))

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
    norm=TwoSlopeNorm(vcenter=0, vmin=-0.4, vmax=0.4),
    # norm=TwoSlopeNorm(vcenter=0, vmin=np.nanmin(aep_values), vmax=np.nanmax(aep_values)),
    aspect="equal",
    alpha=0.7,
)
for wf_idx, wf in enumerate(wind_farms):
    for aep_idx, aep_model in enumerate(aep_models):
        value = normalized_tables[wf].loc[aep_model, "normalized_AEP"]
        ax.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black")
plt.colorbar(im, ax=ax, ticks=np.arange(-1, 1, 0.1), format="%.2f", extend="both")
plt.xticks(range(len(aep_models)), aep_models, rotation=45)
plt.yticks(range(len(wind_farms)), wind_farm_names)
if norm == "max":
    plt.title(f"Normalized AEP - AEP/AEP_ref per farm \n (wake model: {wake_model})")
else:
    plt.title(f"AEP difference with respect to reference (360 WD) in percentage \n (wake model: {wake_model})")
plt.tight_layout()
plt.show()
plt.savefig(f"AEP_heatmap_{wake_model}.png", dpi=300)

# --- Plot heat map for optimization time ---
fig, ax = plt.subplots(figsize=(8, 4))
time_values = np.array([df["normalized_time"].values for df in normalized_tables.values()])
im = ax.imshow(
    time_values,
    cmap="RdYlGn_r",
    # norm=TwoSlopeNorm(vcenter=0, vmin=np.nanmin(time_values), vmax=np.nanmax(time_values)),
    norm=TwoSlopeNorm(vcenter=0, vmin=-100, vmax=100),
    aspect="auto",
    alpha=0.7,
)
for wf_idx, wf in enumerate(wind_farms):
    for aep_idx, aep_model in enumerate(aep_models):
        value = normalized_tables[wf].loc[aep_model, "normalized_time"]
        ax.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black")
plt.colorbar(im, ax=ax, ticks=[-100, -75, -50, -25, 0, 25, 50, 75, 100], format="%.0f", extend="max")
plt.xticks(range(len(aep_models)), aep_models, rotation=45)
plt.yticks(range(len(wind_farms)), wind_farm_names)
if norm == "max":
    plt.title(f"Normalized Optimization Time - Time/Time_ref per farm \n (wake model: {wake_model})")
else:
    plt.title(f"Optimization time difference with respect to reference (360 WD) in percentage \n (wake model: {wake_model})")
plt.tight_layout()
plt.show()
plt.savefig(f"time_heatmap_{wake_model}.png", dpi=300)