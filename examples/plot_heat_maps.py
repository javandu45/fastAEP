# %%

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})


# %%
wake_model = "TurbOPark"  # "NOJ", "Gaussian", "TurbOPark"

aep_models = ["360_WD", "72_WD", "Average_WS", "Uniform_CT", "FLOWERS", "BQ", "RQ", "SGD"]
aep_model_names = ["360 WD", "72 WD", "Average WS", "Uniform CT", "FLOWERS", "BQ", "RQ", "SGD"]


# C: Convex, NC: Non-convex, DC: distancing constraints
wind_farms = ["Hornsea_Project_3_HOW03", "Sofia", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea"]
wind_farm_names = ["Hornsea 3 \n (231 WT, C)", "Sofia \n (100 WT, C, DC)", "Thor \n (72 WT, NC)", "Hornsea 2 \n (55 WT, NC, DC)"]


results_dir = Path(__file__).parent.parent / "results" / "optimization" / "Iteration_3_4WF_30starts"
n_starts = 30

# Normalize with respect to: "360_WD" or "max"
norm = "360_WD"

def obtain_tables(wake_model):

    aep_models_read = ["360_WD", "72_WD", "Average_WS", "Uniform_CT", "FLOWERS", "BQ", "RQ", "SGD"]

    # --- Build median tables per wind farm ---
    tables = {}
    for wf in wind_farms:
        rows = []
        h5_path = results_dir / f"windfarm_{wf}.h5"
        with h5py.File(h5_path, "r") as f:
            for aep_model in aep_models_read:
                aeps, times = [], []
                for start_id in range(n_starts):
                    group_key = f"{aep_model}/{wake_model}/start_{start_id}"
                    if group_key not in f:
                        continue  # No more starts
                    grp = f[group_key]
                    attrs = grp.attrs
                    aeps.append(attrs["aep_final"])
                    times.append(attrs["time"])

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

    # for wf, df in normalized_tables.items():
    #     print(f"\n=== {wf} (wake model: {wake_model}) - Normalized ===")
    #     print(df.to_string(float_format="%.3f"))

    return normalized_tables


normalized_tables = obtain_tables(wake_model)

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
plt.xticks(range(len(aep_models)), aep_model_names, rotation=45)
plt.yticks(range(len(wind_farms)), wind_farm_names)
if norm == "max":
    plt.title(f"Normalized AEP - AEP/AEP_ref per farm \n (wake model: {wake_model})")
else:
    plt.title(f"AEP difference with respect to reference (360 WD) in percentage \n (wake model: {wake_model})")
plt.tight_layout()
plt.show()
# plt.savefig(f"AEP_heatmap_{wake_model}.png", dpi=300)

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
plt.xticks(range(len(aep_models)), aep_model_names, rotation=45)
plt.yticks(range(len(wind_farms)), wind_farm_names)
if norm == "max":
    plt.title(f"Normalized Optimization Time - Time/Time_ref per farm \n (wake model: {wake_model})")
else:
    plt.title(f"Optimization time difference with respect to reference (360 WD) in percentage \n (wake model: {wake_model})")
plt.tight_layout()
plt.show()
# plt.savefig(f"time_heatmap_{wake_model}.png", dpi=300)

# %% 
# --- Plot all wake models together in a single figure ---
wake_models = ["NOJ", "Gaussian", "TurbOPark"]
wake_model_names = ["NO Jensen", "Gaussian Bastankhah", "TurbOPark"]

title_size = 18
y_label_size = 14
x_label_size = 14
values_size = 13
wake_model_size = 16

### Final optimized AEPs

# Fixed layout geometry (identical for both figures below) so that suptitle/
# colorbar label text of differing length cannot change the heatmap panel
# proportions between figures.
LAYOUT_RECT = dict(top=0.90, bottom=0.14, left=0.18, right=0.95, hspace=0.35)
CBAR_RECT = [0.18, 0.045, 0.77, 0.018]  # [left, bottom, width, height]
SUPTITLE_Y = 0.965

fig, axs = plt.subplots(len(wake_models), 1, figsize=(8, 11), sharex=True)
fig.subplots_adjust(**LAYOUT_RECT)

last_im = None

for i, wake_model in enumerate(wake_models):
    ax = axs[i]
    normalized_tables = obtain_tables(wake_model)

    # --- Plot heat map for optimized AEP ---
    aep_values = np.array([df["normalized_AEP"].values for df in normalized_tables.values()])
    im = ax.imshow(
        aep_values,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vcenter=0, vmin=-0.4, vmax=0.4),
        aspect="auto",
        alpha=0.7,
    )
    last_im = im
    for wf_idx, wf in enumerate(wind_farms):
        for aep_idx, aep_model in enumerate(aep_models):
            value = normalized_tables[wf].loc[aep_model, "normalized_AEP"]
            ax.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=values_size)
    # Borders instead of gridlines
    for k in range(len(wind_farms)):
        ax.axhline(k - 0.5, color="white", lw=1.5)
    for j in range(len(aep_models)):
        ax.axvline(j - 0.5, color="white", lw=1.5)
    ax.set_xticks(range(len(aep_models)))
    ax.set_xticklabels(aep_model_names, rotation=30, ha="center", fontsize=x_label_size)
    ax.set_yticks(range(len(wind_farms)))
    ax.set_yticklabels(wind_farm_names, fontsize=y_label_size)
    if norm == "max":
        ax.set_title(f"Normalized AEP - AEP/AEP_ref per farm \n (wake model: {wake_model})")
    else:
        ax.set_title(f"{wake_model_names[i]}", fontsize=wake_model_size)

cbar_ax = fig.add_axes(CBAR_RECT)
cb = fig.colorbar(
    last_im,
    cax=cbar_ax,
    orientation="horizontal",
    ticks=np.arange(-1, 1, 0.1),
    format="%.2f",
    extend="both",
)
cb.set_label("AEP difference, $100 \.  AEP/AEP_{360 \. WD}$ (%)", fontsize=y_label_size)
fig.suptitle("AEP difference with reference (360 WD) in %", fontsize=title_size, y=SUPTITLE_Y)
plt.show()

### Optimization times

fig, axs = plt.subplots(len(wake_models), 1, figsize=(8, 11), sharex=True)
fig.subplots_adjust(**LAYOUT_RECT)

last_im = None

for i, wake_model in enumerate(wake_models):
    ax = axs[i]
    normalized_tables = obtain_tables(wake_model)

    # --- Plot heat map for optimization time ---
    aep_values = np.array([df["normalized_time"].values for df in normalized_tables.values()])
    im = ax.imshow(
        aep_values,
        cmap="RdYlGn_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-100, vmax=100),
        aspect="auto",
        alpha=0.7,
    )
    last_im = im
    for wf_idx, wf in enumerate(wind_farms):
        for aep_idx, aep_model in enumerate(aep_models):
            value = normalized_tables[wf].loc[aep_model, "normalized_time"]
            ax.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=values_size)
    # Borders instead of gridlines
    for k in range(len(wind_farms)):
        ax.axhline(k - 0.5, color="white", lw=1.5)
    for j in range(len(aep_models)):
        ax.axvline(j - 0.5, color="white", lw=1.5)
    ax.set_xticks(range(len(aep_models)))
    ax.set_xticklabels(aep_model_names, rotation=30, ha="center", fontsize=x_label_size)
    ax.set_yticks(range(len(wind_farms)))
    ax.set_yticklabels(wind_farm_names, fontsize=y_label_size)
    if norm == "max":
        ax.set_title(f"Normalized Optimization Time - Time/Time_ref per farm \n (wake model: {wake_model})")
    else:
        ax.set_title(f"{wake_model_names[i]}", fontsize=wake_model_size)

cbar_ax = fig.add_axes(CBAR_RECT)
cb = fig.colorbar(
    last_im,
    cax=cbar_ax,
    orientation="horizontal",
    ticks=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
    format="%.0f",
    extend="max",
)
cb.set_label("Optimization time difference, $100 \.  t/t_{360 \. WD}$ (%)", fontsize=y_label_size)

fig.suptitle("Optimization time difference with reference (360 WD) in %", fontsize=title_size, y=SUPTITLE_Y)
plt.show()


# %% Combined figure: AEP (left) and Optimization Time (right)

wake_models = ["NOJ", "Gaussian", "TurbOPark"]

aep_models = ["360_WD", "72_WD", "Average_WS", "Uniform_CT", "FLOWERS", "BQ", "RQ", "SGD"]
aep_model_names = ["Baseline", "Coarse WD", "Average WS", "Uniform $C_T$", "FLOWERS", "BQ", "OWQ", "SGD"]

# C: Convex, NC: Non-convex, DC: distancing constraints
wind_farms = ["Hornsea_Project_3_HOW03", "Sofia", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea"]
wind_farm_names = ["Hornsea 3 \n (231 WT, C)", "Sofia \n (100 WT, C, DC)", "Thor \n (72 WT, NC)", "Hornsea 2 \n (55 WT, NC, DC)"]


LAYOUT_RECT_COMBINED = dict(top=0.90, bottom=0.14, left=0.12, right=0.95, wspace=0.1, hspace=0.2)
CBAR_RECT_AEP = [0.12, 0.045, 0.35, 0.018]  # [left, bottom, width, height]
CBAR_RECT_TIME = [0.58, 0.045, 0.35, 0.018]  # [left, bottom, width, height]

fig, axs = plt.subplots(len(wake_models), 2, figsize=(14, 11), sharex=True, sharey="col")
fig.subplots_adjust(**LAYOUT_RECT_COMBINED)

last_im_aep = None
last_im_time = None

for i, wake_model in enumerate(wake_models):
    normalized_tables = obtain_tables(wake_model)
    
    # Left subplot: AEP
    ax_aep = axs[i, 0]
    aep_values = np.array([df["normalized_AEP"].values for df in normalized_tables.values()])
    im_aep = ax_aep.imshow(
        aep_values,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1),
        aspect="auto",
        alpha=0.7,
    )
    last_im_aep = im_aep
    for wf_idx, wf in enumerate(wind_farms):
        for aep_idx, aep_model in enumerate(aep_models):
            value = normalized_tables[wf].loc[aep_model, "normalized_AEP"]
            ax_aep.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=values_size)
    # Borders instead of gridlines
    for k in range(len(wind_farms)):
        ax_aep.axhline(k - 0.5, color="white", lw=1.5)
    for j in range(len(aep_models)):
        ax_aep.axvline(j - 0.5, color="white", lw=1.5)
    ax_aep.set_xticks(range(len(aep_models)))
    ax_aep.set_xticklabels(aep_model_names, rotation=30, ha="center", fontsize=x_label_size)
    ax_aep.set_yticks(range(len(wind_farms)))
    ax_aep.set_yticklabels(wind_farm_names, fontsize=y_label_size)
    ax_aep.set_title(f"{wake_model_names[i]} - AEP", fontsize=wake_model_size)
    
    # Right subplot: Optimization Time
    ax_time = axs[i, 1]
    time_values = np.array([df["normalized_time"].values for df in normalized_tables.values()])
    im_time = ax_time.imshow(
        time_values,
        cmap="RdYlGn_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-100, vmax=100),
        aspect="auto",
        alpha=0.7,
    )
    last_im_time = im_time
    for wf_idx, wf in enumerate(wind_farms):
        for aep_idx, aep_model in enumerate(aep_models):
            value = normalized_tables[wf].loc[aep_model, "normalized_time"]
            ax_time.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=values_size)
    # Borders instead of gridlines
    for k in range(len(wind_farms)):
        ax_time.axhline(k - 0.5, color="white", lw=1.5)
    for j in range(len(aep_models)):
        ax_time.axvline(j - 0.5, color="white", lw=1.5)
    ax_time.set_xticks(range(len(aep_models)))
    ax_time.set_xticklabels(aep_model_names, rotation=30, ha="center", fontsize=x_label_size)
    ax_time.set_yticks(range(len(wind_farms)))
    if i == 0:
        ax_time.set_yticklabels([])
    ax_time.set_title(f"{wake_model_names[i]} - Optimization Time", fontsize=wake_model_size)

# Colorbar for AEP
cbar_ax_aep = fig.add_axes(CBAR_RECT_AEP)
cb_aep = fig.colorbar(
    last_im_aep,
    cax=cbar_ax_aep,
    orientation="horizontal",
    ticks=np.arange(-1, 1.25, 0.25),
    format="%.2f",
    extend="min",
)
cb_aep.set_label("AEP difference (%)", fontsize=y_label_size)

# Colorbar for Time
cbar_ax_time = fig.add_axes(CBAR_RECT_TIME)
cb_time = fig.colorbar(
    last_im_time,
    cax=cbar_ax_time,
    orientation="horizontal",
    ticks=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
    format="%.0f",
    extend="max",
)
cb_time.set_label("Optimization time difference (%)", fontsize=y_label_size)

fig.suptitle("AEP and Optimization Time Comparison with Baseline", fontsize=title_size, y=SUPTITLE_Y)
plt.show()


# %% Combined figure: AEP (left) and Optimization Time (right) - Not including 360 WD
wake_models = ["NOJ", "Gaussian", "TurbOPark"]
wake_model_names = ["NOJ", "BPA", "NTP"]

aep_models = ["72_WD", "Average_WS", "Uniform_CT", "FLOWERS", "BQ", "RQ", "SGD"]
aep_model_names = ["Coarse WD", "Average WS", "Uniform $C_T$", "FLOWERS", "BQ", "OWQ", "SGD"]

# C: Convex, NC: Non-convex, DC: distancing constraints
wind_farms = ["Hornsea_Project_3_HOW03", "Sofia", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea"]
wind_farm_names = ["Hornsea 3 \n (231 WT, C)", "Sofia \n (100 WT, C, DC)", "Thor \n (72 WT, NC)", "Hornsea 2 \n (55 WT, NC, DC)"]


LAYOUT_RECT_COMBINED = dict(top=0.90, bottom=0.14, left=0.12, right=0.95, wspace=0.1, hspace=0.2)
CBAR_RECT_AEP = [0.12, 0.045, 0.35, 0.018]  # [left, bottom, width, height]
CBAR_RECT_TIME = [0.58, 0.045, 0.35, 0.018]  # [left, bottom, width, height]

fig, axs = plt.subplots(len(wake_models), 2, figsize=(14, 11), sharex=True, sharey="col")
fig.subplots_adjust(**LAYOUT_RECT_COMBINED)

last_im_aep = None
last_im_time = None

for i, wake_model in enumerate(wake_models):
    normalized_tables = obtain_tables(wake_model)

    # Left subplot: AEP
    ax_aep = axs[i, 0]
    aep_values = np.array([df.loc[aep_models, "normalized_AEP"].values for df in normalized_tables.values()])
    im_aep = ax_aep.imshow(
        aep_values,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1),
        aspect="auto",
        alpha=0.6,
    )
    last_im_aep = im_aep
    for wf_idx, wf in enumerate(wind_farms):
        for aep_idx, aep_model in enumerate(aep_models):
            value = normalized_tables[wf].loc[aep_model, "normalized_AEP"]
            ax_aep.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=values_size)
    for k in range(len(wind_farms)):
        ax_aep.axhline(k - 0.5, color="white", lw=1.5)
    for j in range(len(aep_models)):
        ax_aep.axvline(j - 0.5, color="white", lw=1.5)
    ax_aep.set_xticks(range(len(aep_models)))
    ax_aep.set_xticklabels(aep_model_names, rotation=30, ha="center", fontsize=x_label_size)
    ax_aep.set_yticks(range(len(wind_farms)))
    ax_aep.set_yticklabels(wind_farm_names, fontsize=y_label_size)
    ax_aep.set_title(f"{wake_model_names[i]} - AEP", fontsize=wake_model_size)

    # Right subplot: Optimization Time
    ax_time = axs[i, 1]
    time_values = np.array([df.loc[aep_models, "normalized_time"].values for df in normalized_tables.values()])
    im_time = ax_time.imshow(
        time_values,
        cmap="RdYlGn_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-100, vmax=100),
        aspect="auto",
        alpha=0.6,
    )
    last_im_time = im_time
    for wf_idx, wf in enumerate(wind_farms):
        for aep_idx, aep_model in enumerate(aep_models):
            value = normalized_tables[wf].loc[aep_model, "normalized_time"]
            ax_time.text(aep_idx, wf_idx, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=values_size)
    for k in range(len(wind_farms)):
        ax_time.axhline(k - 0.5, color="white", lw=1.5)
    for j in range(len(aep_models)):
        ax_time.axvline(j - 0.5, color="white", lw=1.5)
    ax_time.set_xticks(range(len(aep_models)))
    ax_time.set_xticklabels(aep_model_names, rotation=30, ha="center", fontsize=x_label_size)
    ax_time.set_yticks(range(len(wind_farms)))
    ax_time.set_yticklabels([], fontsize=y_label_size)
    ax_time.set_title(f"{wake_model_names[i]} - Optimization Time", fontsize=wake_model_size)

# Colorbar for AEP
cbar_ax_aep = fig.add_axes(CBAR_RECT_AEP)
cb_aep = fig.colorbar(
    last_im_aep,
    cax=cbar_ax_aep,
    orientation="horizontal",
    ticks=np.arange(-1, 1.25, 0.25),
    format="%.2f",
    extend="min",
)
cb_aep.set_label("AEP difference (%)", fontsize=y_label_size)
cb_aep.ax.tick_params(labelsize=12)

# Colorbar for Time
cbar_ax_time = fig.add_axes(CBAR_RECT_TIME)
cb_time = fig.colorbar(
    last_im_time,
    cax=cbar_ax_time,
    orientation="horizontal",
    ticks=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
    format="%.0f",
    extend="max",
)
cb_time.set_label("Optimization time difference (%)", fontsize=y_label_size)
cb_time.ax.tick_params(labelsize=12)

fig.suptitle("AEP and Optimization Time Comparison with Baseline", fontsize=title_size, y=SUPTITLE_Y)
plt.savefig(f"AEP_time_heatmap.pdf", dpi=300, bbox_inches="tight")
plt.show()