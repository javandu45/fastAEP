import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

wake_model = "NOJ"
wind_farms = ["East_Anglia_TWO", "Sofia", "Thor"]
aep_models = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "BQ", "RQ", "Uniform_CT"]

results_dir = Path("results/optimization")
n_starts = 30

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
    max_aep = df["median_AEP_GWh"].max()
    min_aep = df["median_AEP_GWh"].min()
    max_time = df["median_time_s"].max()
    min_time = df["median_time_s"].min()
    df_norm = df.copy()
    df_norm["normalized_AEP"] = (df["median_AEP_GWh"] - min_aep) / (max_aep - min_aep)
    df_norm["normalized_time"] = df["median_time_s"] / min_time
    normalized_tables[wf] = df_norm

for wf, df in normalized_tables.items():
    print(f"\n=== {wf} (wake model: {wake_model}) - Normalized ===")
    print(df.to_string(float_format="%.3f"))

# --- Plot bubble plots with y-axis the wind farms, x-axis the normalized time, and bubble size the normalized AEP ---
colors = plt.cm.tab10(np.linspace(0, 1, len(aep_models)))
color_map = dict(zip(aep_models, colors))

fig, ax = plt.subplots(figsize=(12, 5))

for wf_idx, wf in enumerate(wind_farms):
    df = normalized_tables[wf]
    ref_size = (df["normalized_AEP"].max() + 0.1) * 50
    for aep_model in aep_models:
        x = df.loc[aep_model, "normalized_time"]
        y = wf_idx
        size = (df.loc[aep_model, "normalized_AEP"] + 0.1) * 50
        ax.scatter(x, y, s=size * 20, color=color_map[aep_model],
                   alpha=0.7, edgecolors="k", linewidths=0.5,
                   label=aep_model if wf_idx == 0 else "_nolegend_")
        # Add an empty reference bubble showing the maximum AEP size for this wind farm
        ax.scatter(x, y, s=ref_size * 20, marker="o",
                   facecolors="none", edgecolors="k", linewidths=1.0,
                   label="_nolegend_")

color_handles = [
    plt.Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markersize=8,
        markerfacecolor=color_map[aep_model],
        markeredgecolor="k",
        alpha=0.7,
        label=aep_model,
    )
    for aep_model in aep_models
]

# Include empty bubble legend for reference size
legend_ref_size = max(
    (df["normalized_AEP"].max() + 0.1) * 50 for df in normalized_tables.values()
) * 20
empty_bubble_handle = plt.Line2D(
    [0], [0],
    marker="o",
    linestyle="",
    markersize=np.sqrt(legend_ref_size),
    markerfacecolor="none",
    markeredgecolor="k",
    linewidth=1.0,
    label="Max AEP for farm",
)

ax.set_yticks(range(len(wind_farms)))
ax.set_yticklabels(wind_farms)
ax.set_xticks(np.array([1, 10, 20, 30, 40, 50, 60, 70]))
ax.set_xlabel("Normalized Time (t/t_min)")
ax.set_ylabel("Wind farm")
ax.set_ylim(-0.5, len(wind_farms) - 0.5)
ax.set_title(f"AEP vs. Time — Wake model: {wake_model}\n(bubble size = AEP / AEP_max for each wind farm)")
legend_methods = ax.legend(handles=color_handles, title="AEP method",
                           bbox_to_anchor=(1.01, 1), loc="upper left")
ax.add_artist(legend_methods)
ax.legend(handles=[empty_bubble_handle],
          bbox_to_anchor=(1.01, 0.42), loc="upper left", borderpad=1.75, handletextpad=2)
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
