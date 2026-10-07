
# Code used to generate several plots for the paper
# A comparison of efficient AEP models for Wind Farm Layout Optimization

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from FAST_AEP.BQ import bayesian_quadrature
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from FAST_AEP.FLOWERS import NOJ_flowers, gaussian_flowers, TurbOPark_flowers
from FAST_AEP.utils import generic_site, build_aep_model, turbine_generator
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

# wind_farm = "Thor"

# turbine = turbine_generator(wind_farm)

# D = turbine.diameter()  # Turbine diameter in meters
# fontsize = 14

# def plot_9th_turbine_aep_impact(proxy, ax, grid_resolution=50):
#     """
#     Plot AEP changes when adding a 9th turbine to 8 turbines in a square layout.

#     Args:
#         proxy: The wind farm proxy object with calculate_aep method
#         ax: Matplotlib axes object to plot on
#         grid_resolution: Number of points along each axis for the grid

#     Returns:
#         contourf: The contourf object for creating a shared colorbar
#     """

#     # Define 8 turbines in a square layout
#     base_layout = np.array([
#         [0, 0],
#         [7*D, 0],
#         [14*D, 0],
#         [0, 7*D],
#         [14*D, 7*D],
#         [0, 14*D],
#         [7*D, 14*D],
#         [14*D, 14*D]
#     ])

#     # Calculate baseline AEP with 8 turbines
#     baseline_aep = proxy.aep(base_layout[:, 0], base_layout[:, 1])

#     # Get turbine diameter and calculate minimum spacing (2D)
#     turbine_diameter = D
#     min_spacing = 2 * turbine_diameter

#     # Create grid for 9th turbine positions
#     x_range = np.linspace(0*D, 14*D, grid_resolution)
#     y_range = np.linspace(0*D, 14*D, grid_resolution)
#     X, Y = np.meshgrid(x_range, y_range)

#     # Calculate AEP for each position of 9th turbine
#     aep_values = np.zeros_like(X)
#     for i in range(grid_resolution):
#         for j in range(grid_resolution):
#             x_9th = X[i, j]
#             y_9th = Y[i, j]

#             # Check minimum spacing constraint
#             distances = np.sqrt((base_layout[:, 0] - x_9th)**2 + (base_layout[:, 1] - y_9th)**2)
#             if np.any(distances < min_spacing):
#                 aep_values[i, j] = np.nan  # Invalid position
#                 continue

#             x_all = np.append(base_layout[:, 0], x_9th)
#             y_all = np.append(base_layout[:, 1], y_9th)
#             aep_values[i, j] = proxy.aep(x_all, y_all) - baseline_aep

#     max_aep = np.nanmax(aep_values)
#     if max_aep > 0:
#         aep_values = aep_values / max_aep

#     # Create plot on the provided axes
#     contourf = ax.contourf(X/D, Y/D, aep_values, levels=50, cmap='RdYlGn')
#     contour = ax.contour(X/D, Y/D, aep_values, levels=50, colors='black', linewidths=0.5, alpha=1)

#     # Plot existing 8 turbines (normalized)
#     ax.scatter(base_layout[:, 0]/D, base_layout[:, 1]/D,
#                 c='black', s=500, marker='o',
#                 edgecolors='black', linewidths=5,
#                 label='Existing Turbines', zorder=5)

#     ax.set_xlabel('X/$D_{WT}$', fontsize=fontsize)
#     ax.set_ylabel('Y/$D_{WT}$', fontsize=fontsize)
#     ax.grid(True, alpha=0.3)
#     ax.set_xticks([0, 7, 14])
#     ax.set_yticks([0, 7, 14])
#     ax.tick_params(labelsize=fontsize)
#     ax.set_aspect('equal')
#     ax.set_xlim(0, 14)
#     ax.set_ylim(0, 14)

#     return contourf

# # %%

# site = generic_site(wind_farm="Thor")
# aep_models = ["360_WD", "72_WD", "Average_WS", "Uniform_CT", "FLOWERS", "BQ", "RQ"]
# wake_model = "TurbOPark"
# proxies = [build_aep_model(aep_method=aep_model, wind_farm="Thor", deficit_model=wake_model) for aep_model in aep_models]
# names = ["360 WD", "72 WD", "Average WS", "Uniform CT", "FLOWERS", "BQ", "RQ"]

# # Create figure with 7 subplots
# fig, axes = plt.subplots(4, 2, figsize=(13, 22), dpi=200)
# axes = axes.ravel()

# contourf = None
# for i, proxy in enumerate(proxies):
#     # Plot AEP impact for each proxy and get contourf object
#     print("Plotting for proxy:", names[i])
#     contourf = plot_9th_turbine_aep_impact(proxy, axes[i], grid_resolution=100)
#     axes[i].set_title(f"Proxy: {names[i]}", fontsize=fontsize+2)

# # Hide the unused subplot
# axes[len(proxies)].axis('off')

# # Get legend from first subplot
# handles, labels = axes[0].get_legend_handles_labels()

# # Create shared colorbar
# plt.tight_layout()
# fig.subplots_adjust(right=0.9)
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(contourf, cax=cbar_ax, format="%.3f")
# cbar.set_label('Normalized AEP ($AEP/AEP_{max}$)', fontsize=fontsize)
# cbar.ax.tick_params(labelsize=fontsize)
# fig.legend(handles, labels, fontsize=fontsize, ncol=1, loc='lower center', bbox_to_anchor=(0.5, -0.03))
# plt.savefig(f"9th_turbine_AEP_impact_{wake_model}.png", dpi=300, bbox_inches='tight')
# plt.show()

########################################################################
# %% --- PLOT AEP BENCHMARKING ---

results_path = str(Path(__file__).parent.parent / "results" / "benchmarking")
# results_path = "results/benchmarking"

# #############
# AEP_VALUES
# Read results
aeps_noj = pd.read_csv(results_path + "/benchmark_aep_noj_2.csv")
aeps_gaussian = pd.read_csv(results_path + "/benchmark_aep_gaussian_2.csv")
aeps_turbopark = pd.read_csv(results_path + "/benchmark_aep_turbopark_2.csv")

turbine_ranges = [25, 50, 75, 100, 150, 200, 250, 300]
aep_methods = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "BQ", "RQ", "Uniform_CT"]

# NOJ
plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    y_values = 100 * (np.array(aeps_noj[aep_method]) - np.array(aeps_noj["360_WD"])) / np.array(aeps_noj["360_WD"])
    plt.plot(turbine_ranges, y_values, label=aep_method, marker="o", linewidth=2)
plt.xlabel("Number of turbines")
plt.ylabel("AEP difference relative to 360 WD [%]")
plt.title("AEP for different methods with NOJ wake model")
plt.legend()
plt.grid()
# plt.savefig(results_path + "/benchmark_aep_noj.png", dpi=300)
plt.show()

# Gaussian
plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    y_values = 100 * (np.array(aeps_gaussian[aep_method]) - np.array(aeps_gaussian["360_WD"])) / np.array(aeps_gaussian["360_WD"])
    plt.plot(turbine_ranges, y_values, label=aep_method, marker="o", linewidth=2)
plt.xlabel("Number of turbines")
plt.ylabel("AEP difference relative to 360 WD [%]")
plt.title("AEP for different methods with Gaussian wake model")
plt.legend()
plt.grid()
# plt.savefig(results_path + "/benchmark_aep_gaussian.png", dpi=300)
plt.show()

# TurbOPark
plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    y_values = 100 * (np.array(aeps_turbopark[aep_method]) - np.array(aeps_turbopark["360_WD"])) / np.array(aeps_turbopark["360_WD"])
    plt.plot(turbine_ranges, y_values, label=aep_method, marker="o", linewidth=2)
plt.xlabel("Number of turbines")
plt.ylabel("AEP difference relative to 360 WD [%]")
plt.title("AEP for different methods with TurbOPark wake model")
plt.legend()
plt.grid()
# plt.savefig(results_path + "/benchmark_aep_turbopark.png", dpi=300)
plt.show()

# #############
# TIMES VALUES
times_noj = pd.read_csv(results_path + "/benchmark_runtime_noj.csv")
times_gaussian = pd.read_csv(results_path + "/benchmark_runtime_gaussian.csv")
times_turbopark = pd.read_csv(results_path + "/benchmark_runtime_turbopark.csv")

# NOJ
plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    y_values = np.array(times_noj[aep_method])
    plt.plot(turbine_ranges, y_values, label=aep_method, marker="o", linewidth=2)
plt.xlabel("Number of turbines")
plt.ylabel("Computation time [s]")
plt.title("Computation time for different methods with NOJ wake model")
plt.legend()
plt.grid()
# plt.savefig(results_path + "/benchmark_runtime_noj.png", dpi=300)
plt.show()

# Gaussian
plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    y_values = np.array(times_gaussian[aep_method])
    plt.plot(turbine_ranges, y_values, label=aep_method, marker="o", linewidth=2)
plt.xlabel("Number of turbines")
plt.ylabel("Computation time [s]")
plt.title("Computation time for different methods with Gaussian wake model")
plt.legend()
plt.grid()
# plt.savefig(results_path + "/benchmark_runtime_gaussian.png", dpi=300)
plt.show()

# TurbOPark
plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    y_values = np.array(times_turbopark[aep_method])
    plt.plot(turbine_ranges, y_values, label=aep_method, marker="o", linewidth=2)
plt.xlabel("Number of turbines")
plt.ylabel("Computation time [s]")
plt.title("Computation time for different methods with TurbOPark wake model")
plt.legend()
plt.grid()
# plt.savefig(results_path + "/benchmark_runtime_turbopark.png", dpi=300)
plt.show()

###########################################################################
# %% --- PLOT AEP AND TIME BENCHMARKING IN A SINGLE PLOT ---

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

results_path = str(Path(__file__).parent.parent / "results" / "benchmarking")

use_parallelization = True
cpu_hours = True

n_cpus = [8, 8, 8, 8, 1, 8, 8, 1]

# #############
# AEP_VALUES
# Read results
aeps_noj = pd.read_csv(results_path + "/benchmark_aep_noj_2.csv")
aeps_gaussian = pd.read_csv(results_path + "/benchmark_aep_gaussian_2.csv")
aeps_turbopark = pd.read_csv(results_path + "/benchmark_aep_turbopark_2.csv")

# aeps_noj = pd.read_csv(results_path + "/benchmark_runtime_noj_no_parallel_g.csv")
# aeps_gaussian = pd.read_csv(results_path + "/benchmark_runtime_gaussian_no_parallel_g.csv")
# aeps_turbopark = pd.read_csv(results_path + "/benchmark_runtime_turbopark_no_parallel_g.csv")

# #############
# TIMES VALUES
if use_parallelization:
    times_noj = pd.read_csv(results_path + "/benchmark_runtime_noj_g.csv")
    times_gaussian = pd.read_csv(results_path + "/benchmark_runtime_gaussian_g.csv")
    times_turbopark = pd.read_csv(results_path + "/benchmark_runtime_turbopark_g.csv")
else:
    times_noj = pd.read_csv(results_path + "/benchmark_runtime_noj_no_parallel.csv")
    times_gaussian = pd.read_csv(results_path + "/benchmark_runtime_gaussian_no_parallel.csv")
    times_turbopark = pd.read_csv(results_path + "/benchmark_runtime_turbopark_no_parallel.csv")

turbine_ranges = [25, 50, 75, 100, 150, 200, 250, 300]
aep_methods = ["360_WD", "72_WD", "Average_WS", "Uniform_CT", "FLOWERS", "BQ", "RQ"]
aep_method_labels = ["Baseline", "Coarse WD", "Average WS", "Uniform $C_T$", "FLOWERS", "BQ", "OWQ"]


if cpu_hours:
    for i, method in enumerate(aep_methods):
        cpu_factor = n_cpus[i]
        for data in [times_noj, times_gaussian, times_turbopark]:
            if method in data.columns:
                data[method] = pd.to_numeric(data[method], errors="coerce") * cpu_factor

markers = ["", "o", "s", "^", "o", "s", "^", "D"]
line_styles = ["--", "-", "-", "-", ":", ":", ":", ":"]

def plot_benchmark(ax, data, title, ylabel, xlabel):
    for i, aep_method in enumerate(aep_methods):
        y_values = 100 * (np.array(data[aep_method]) - np.array(data["360_WD"])) / np.array(data["360_WD"])
        ax.plot(turbine_ranges, y_values, label=aep_method_labels[i], marker=markers[i], linewidth=2, linestyle=line_styles[i])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.grid()

# fig, axs = plt.subplots(1, 3, figsize=(12, 6), dpi=300, sharex=True, sharey="row")
fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300, sharex=True, sharey="row")
axs = axs.ravel()
for i, (data, title, ylabel, xlabel) in enumerate(zip(
    # [aeps_noj, aeps_gaussian, aeps_turbopark, times_noj, times_gaussian, times_turbopark],
    [aeps_noj, aeps_gaussian, aeps_turbopark],
    # [times_noj, times_gaussian, times_turbopark],
    # ["NOJ", "BPA", "NTP", "", "", ""],
    ["NOJ", "BPA", "NTP"],
    # ["AEP difference [%]", "", "", "Time difference [%]", "", ""],
    ["AEP difference [%]", "", ""],
    # ["Time difference [%]", "", ""],
    # ["", "", "", "Number of turbines", "Number of turbines", "Number of turbines"]
    ["Number of turbines", "Number of turbines", "Number of turbines"]
)):
    plot_benchmark(axs[i], data, title, ylabel, xlabel)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(aep_methods), bbox_to_anchor=(0.5, -0.1), fontsize=12)
plt.tight_layout()

plt.savefig(f"aep_time_benchmarking_2.pdf", dpi=300, bbox_inches="tight")


######################################################################
# %% --- AEP scatter plots for different methods and wake models ---

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

wake_model = "TurbOPark"  # "NOJ", "Gaussian", "TurbOPark"
aep_models = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "Uniform_CT", "BQ", "RQ", "SGD"]

wind_farms = ["Hornsea_Project_3_HOW03", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea", "Sofia"]
wind_farm_names = ["Hornsea 3 \n (231 WT)", "Thor \n (72 WT)", "Hornsea 2 \n (55 WT)", "Sofia \n (100 WT)"]

results_dir = Path(__file__).resolve().parent.parent / "results/optimization/Iteration_3_4WF_30starts"
n_starts = 30

# Normalize with respect to: "360_WD" or "max"
norm = "360_WD"

# --- Build median tables per wind farm ---
tables = {}
for wf in wind_farms:
    rows = []
    h5_path = f"{results_dir}/windfarm_{wf}.h5"
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
                "AEP_GWh_results": aeps,
                "time_s_results": times,
            })
    tables[wf] = pd.DataFrame(rows).set_index("AEP_model")

# %% --- Plot wind farm boundaries ---

import matplotlib.pyplot as plt
import numpy as np
from FAST_AEP.utils import get_limits
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

wind_farms = ["Hornsea_Project_3_HOW03", "Sofia", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea"]
wind_farm_names = ["Hornsea 3 \n (231 WT, C)", "Sofia \n (100 WT, C, DC)", "Thor \n (72 WT, NC)", "Hornsea 2 \n (55 WT, NC, DC)"]

fig, axes = plt.subplots(1, 4, figsize=(12,12))
axes = axes.ravel()


for i, (ax, wf, wf_name) in enumerate(zip(axes, wind_farms, wind_farm_names)):

    if wf == "Hornsea_Project_3_HOW03":
        limits = np.array([[ 460226.33753195, 5962329.18655583],
       [ 486329.9958098 , 5948829.197878  ],
       [ 479155.27816924, 5983568.46029649],
       [ 447351.63299954, 5982707.50183493],
       [ 460226.33753195, 5962329.18655583]])
    else:
        limits = get_limits(wf)

    ax.plot(limits[:, 0], limits[:, 1], "k-", linewidth=2)
    ax.fill(limits[:, 0], limits[:, 1], color="lightgray", alpha=0.5)
    ax.set_title(wf_name, fontsize=10)


    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid()

    if i == 0:
        ax.annotate(
            "",
            xy=(0.10, 0.18),
            xytext=(0.10, 0.06),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8),
        )
        ax.text(0.10, 0.20, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.savefig("wind_farm_boundaries.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %% --- Wind farms wind roses ---

import numpy as np
import matplotlib.pyplot as plt
from FAST_AEP.utils import generic_site

# --- Publication-quality font/style setup (consistent with your other WES figures) ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

wind_farms = ["Hornsea_Project_3_HOW03", "Sofia", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea"]
wind_farm_names = ["Hornsea 3\n(231 WT, C)", "Sofia\n(100 WT, C, DC)", "Thor\n(72 WT, NC)", "Hornsea 2\n(55 WT, NC, DC)"]


# Wind speed bins (m/s) and a perceptually ordered, colorblind-friendlier palette
speed_bins = np.array([0, 5, 10, 15, 20, 25])
colors = ['#2166AC', '#67A9CF', '#F4A582', '#D6604D', '#B2182B']  # blue (calm) -> red (strong)
n_speed_bins = len(colors)

n_sectors = 12
sector_size = 360 // n_sectors

fig, axes = plt.subplots(1, 4, figsize=(10, 3.5), subplot_kw={'projection': 'polar'})
axes = axes.flatten()

for idx, (wf, wf_name) in enumerate(zip(wind_farms, wind_farm_names)):
    ax = axes[idx]

    site = generic_site(wind_farm=wf)
    p = site.ds.Sector_frequency.values
    A = site.ds.Weibull_A.values
    k = site.ds.Weibull_k.values

    freq_by_speed = np.zeros((n_sectors, n_speed_bins))
    for i in range(n_sectors):
        i0, i1 = i * sector_size, (i + 1) * sector_size
        A_sector = np.mean(A[i0:i1])
        k_sector = np.mean(k[i0:i1])
        p_sector = np.mean(p[i0:i1])
        cdf = 1 - np.exp(-(speed_bins / A_sector) ** k_sector)
        freq_by_speed[i, :] = p_sector * np.diff(cdf)

    directions = np.arange(n_sectors) * sector_size + sector_size / 2
    theta = np.deg2rad(directions)

    bottom = np.zeros(n_sectors)
    for speed_idx in range(n_speed_bins):
        ax.bar(
            theta, freq_by_speed[:, speed_idx], width=np.deg2rad(sector_size) * 0.92,
            bottom=bottom,
            label=f'{speed_bins[speed_idx]:.0f}-{speed_bins[speed_idx + 1]:.0f} m/s',
            color=colors[speed_idx], edgecolor='white', linewidth=0.6, zorder=3
        )
        bottom += freq_by_speed[:, speed_idx]

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(wf_name, pad=16, fontsize=11)

    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'], fontsize=10, color='#444444')

    # Light, informative radial grid instead of hiding it entirely
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color='#cccccc', linewidth=0.6, alpha=0.7, zorder=0)
    ax.xaxis.grid(True, color='#cccccc', linewidth=0.6, alpha=0.7, zorder=0)
    ax.spines['polar'].set_color('#999999')
    ax.spines['polar'].set_linewidth(0.8)
    ax.set_facecolor('white')

# Shared legend below the grid
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc='lower center', ncol=n_speed_bins,
    bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9,
    title='Wind speed', title_fontsize=9
)

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig("wind_farm_roses.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %% --- Parallelization comparison ---

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

results_path = str(Path(__file__).parent.parent / "results" / "Parallelization")
aep_models = ["Baseline", "72 WD", "Average WS", "Uniform $C_T$", "BQ", "OWQ"]
deficits = ["NOJ", "Gaussian", "TurbOPark"]

speedup = True
df = pd.read_csv(f"{results_path}/parallelization_g_results_100.csv")

df = df.pivot_table(index=["deficit", "wfm"], columns="n_CPUs", values="time_s")

# Create 7 subplots, one per wind farm model (wfm).
wfms = [w for w in df.index.get_level_values('wfm').unique() if w != "FLOWERS"][:7]
n = len(wfms)
fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=300)
axes = axes.ravel()

for ax, wfm in zip(axes, wfms):
    # select rows for this wfm; rows indexed by deficit, columns by n_CPUs
    sub = df.xs(wfm, level='wfm')
    # plot each deficit as a separate line
    for deficit in deficits:
        if speedup:
            y_values = sub.loc[deficit].values[0]/sub.loc[deficit].values  # Normalize to first CPU count
        else:
            y_values = sub.loc[deficit].values
        ax.plot(sub.columns, y_values, marker='o', label=str(deficit))
    ax.set_title(aep_models[wfms.index(wfm)], fontsize=12)
    ax.set_xlabel('Number of CPUs', fontsize=10)
    ax.set_xscale('linear')
    ax.grid(True, linestyle='--', alpha=0.4)

    ax.set_ylabel('Time (s)', fontsize=10)

for ax in axes[len(wfms):]:
    ax.axis('off')

# Create shared legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))

figures_path = Path(__file__).parent.parent / "figures"

plt.suptitle("Parallelization for 200 WT", fontsize=14)
plt.tight_layout()
# plt.savefig(f"{figures_path}/parallelization_200_wt.png", dpi=300, bbox_inches="tight")
plt.show()


# %% --- Plot clean version of parallelization --- 

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

results_path = str(Path(__file__).parent.parent / "results" / "Parallelization")
aep_models_names = ["Baseline", "Coarse WD", "Average WS", "Uniform $C_T$", "BQ", "OWQ"]
aep_models = ["360 WD", "72 WD", "Average WS", "Uniform CT", "BQ", "RQ"]
deficits = ["NOJ", "Gaussian", "TurbOPark"]
deficits_names = ["NOJ", "BPA", "NTP"]
palette = ["#4C72B0", "#DD8452", "#55A868"]
markers = ["o", "o", "o"]
linestyles = ["-", "--", ":"]

N_tur = 100
speedup = False

df = pd.read_csv(f"{results_path}/parallelization_results_{N_tur}.csv")
df = df.pivot_table(index=["deficit", "wfm"], columns="n_CPUs", values="time_s")

wfms = [w for w in df.index.get_level_values('wfm').unique() if w != "FLOWERS"][:7]

# Smaller overall figure; no shared y since scales differ across wfms
fig, axes = plt.subplots(3, 2, figsize=(4, 4), dpi=300, sharex=True)
axes = axes.ravel()

for ax, wfm in zip(axes, aep_models):
    if wfm == "FLOWERS":
        ax.axis('off')
        continue
    sub = df.xs(wfm, level='wfm')
    for deficit, color, marker in zip(deficits, palette, markers):
        if speedup:
            y_values = sub.loc[deficit].values[0]/sub.loc[deficit].values  # Normalize to first CPU count
        else:
            y_values = sub.loc[deficit].values
        ax.plot(sub.columns, y_values, marker=marker,
                 color=color, markersize=1, linewidth=1.1, label=deficits_names[deficits.index(deficit)],
                 linestyle=linestyles[deficits.index(deficit)])
    ax.set_title(aep_models_names[aep_models.index(wfm)], fontsize=10, pad=2)
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.5)
    ax.tick_params(labelsize=7, pad=1.5)
    ax.set_yticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Cap tick count per axis — this is what keeps small panels legible
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

for ax in axes[len(wfms):]:
    ax.axis('off')

for ax in axes[4:6]:
    ax.set_xlabel('CPUs', fontsize=10, labelpad=2)
for ax in [axes[0], axes[2], axes[4]]:
    ax.set_ylabel('Time', fontsize=10, labelpad=2)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels),
           bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False,
           handletextpad=0.3, columnspacing=1.0)

fig.suptitle(f"AEP parallelization for {N_tur} WT", fontsize=10)

figures_path = Path(__file__).parent.parent / "figures"
plt.tight_layout(rect=[0, 0.06, 1, 0.97], h_pad=0.6, w_pad=0.6)
plt.tight_layout()
plt.savefig(f"parallelization_100_wt.pdf", dpi=300, bbox_inches="tight")
plt.show()


# %% --- Plot clean version of parallelization with gradients ---

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

results_path = str(Path(__file__).parent.parent / "results" / "Parallelization")
aep_models_names = ["Baseline", "Coarse WD", "Average WS", "Uniform $C_T$", "BQ", "OWQ", "SGD"]
aep_models = ["360 WD", "72 WD", "Average WS", "Uniform CT", "BQ", "RQ", "SGD"]
deficits = ["NOJ", "Gaussian", "TurbOPark"]
deficits_names = ["NOJ", "BPA", "NTP"]
palette = ["#4C72B0", "#DD8452", "#55A868"]
markers = ["o", "o", "o"]
linestyles = ["-", "--", ":"]

N_tur = 100
gradients = True
speedup = False

df = pd.read_csv(f"{results_path}/parallelization_g_results_{N_tur}.csv")
df = df.pivot_table(index=["deficit", "wfm"], columns="n_CPUs", values="time_s")
print(df)

wfms = [w for w in df.index.get_level_values('wfm').unique() if w != "FLOWERS"][:8]

# Smaller overall figure; no shared y since scales differ across wfms
fig, axes = plt.subplots(4, 2, figsize=(4, 5), dpi=300, sharex=True)
axes = axes.ravel()

for ax, wfm in zip(axes, aep_models):
    if wfm == "FLOWERS":
        ax.axis('off')
        continue
    sub = df.xs(wfm, level='wfm')
    for deficit, color, marker in zip(deficits, palette, markers):
        if speedup:
            y_values = sub.loc[deficit].values[0]/sub.loc[deficit].values  # Normalize to first CPU count
        else:
            y_values = sub.loc[deficit].values
        ax.plot(sub.columns, y_values, marker=marker,
                 color=color, markersize=1, linewidth=1.1, label=deficits_names[deficits.index(deficit)], linestyle=linestyles[deficits.index(deficit)])
    ax.set_title(aep_models_names[aep_models.index(wfm)], fontsize=10, pad=2)
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.5)
    ax.tick_params(labelsize=7, pad=1.5)
    ax.set_yticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Cap tick count per axis — this is what keeps small panels legible
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

for ax in axes[len(wfms):]:
    ax.axis('off')

for ax in axes[4:6]:
    ax.set_xlabel('CPUs', fontsize=10, labelpad=2)
for ax in [axes[0], axes[2], axes[4], axes[6]]:
    ax.set_ylabel('Time', fontsize=10, labelpad=2)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels),
           bbox_to_anchor=(0.5, -0.1), fontsize=10, frameon=False,
           handletextpad=0.3, columnspacing=1.0)

fig.suptitle(f"Gradient parallelization for {N_tur} WT", fontsize=10)

figures_path = Path(__file__).parent.parent / "figures"
plt.tight_layout(rect=[0, 0.06, 1, 0.97], h_pad=0.6, w_pad=0.6)
plt.tight_layout()

if len(wfms) % 2 == 1:
    last_ax = axes[len(wfms) - 1]
    ref_ax = axes[len(wfms) - 2]
    ref_pos = ref_ax.get_position()
    last_ax.set_position([0.5 - ref_pos.width / 2, 0.05, ref_pos.width, ref_pos.height])
    last_ax.set_xlabel('CPUs', fontsize=10, labelpad=2)

plt.savefig("gradient_parallelization_100_wt.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %% --- Plot number of iterations for different AEP methods and wake models ---

import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np
from pathlib import Path

wind_farms = ["Hornsea_Project_3_HOW03", "Sofia", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea"]
wind_farm_names = ["Hornsea 3", "Sofia", "Thor", "Hornsea 2"]
wake_model = "NOJ"
figures_path = Path(__file__).parent.parent / "figures"
figures_path.mkdir(exist_ok=True)

# Use the same AEP methods as in the previous optimization plots where available.
aep_models = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "Uniform_CT", "BQ", "RQ"]
method_labels = {
    "360_WD": "360 WD",
    "72_WD": "72 WD",
    "Average_WS": "Average WS",
    "FLOWERS": "FLOWERS",
    "Uniform_CT": "Uniform C_T",
    "BQ": "BQ",
    "RQ": "RQ",
}

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
n_starts = 30
method_counts = {wf: [] for wf in wind_farms}
method_names = []

# Load data for all wind farms
for wind_farm in wind_farms:
    results_file = Path(__file__).parent.parent / "results" / "optimization" / "Iteration_3_4WF_30starts"/ f"windfarm_{wind_farm}.h5"
    
    with h5py.File(results_file, "r") as f:
        for aep_model in aep_models:
            counts = []
            for start_id in range(n_starts):
                group_key = f"{aep_model}/{wake_model}/start_{start_id}"
                if group_key not in f:
                    continue
                grp = f[group_key]

                iter_value = None
                for attr_name in ["n_iterations", "iterations", "n_iter", "iter", "n_iter_final", "n_iteration"]:
                    if attr_name in grp.attrs:
                        value = grp.attrs[attr_name]
                        if np.isscalar(value):
                            iter_value = int(value)
                        else:
                            iter_value = int(np.asarray(value).ravel()[0])
                        break

                if iter_value is None:
                    for ds_name in ["n_iterations", "iterations", "n_iter", "iter"]:
                        if ds_name in grp:
                            ds = grp[ds_name]
                            if ds.shape == ():
                                iter_value = int(ds[()])
                            else:
                                iter_value = int(np.asarray(ds).ravel()[0])
                            break

                if iter_value is not None:
                    counts.append(iter_value)

            if counts and wind_farm == wind_farms[0]:
                method_names.append(method_labels.get(aep_model, aep_model))
            
            if counts:
                method_counts[wind_farm].append(counts)

if method_names:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    x_base = np.arange(len(method_names))
    group_width = 0.2
    offsets = np.linspace(-group_width, group_width, len(wind_farms))

    for wf_idx, wind_farm in enumerate(wind_farms):
        for method_idx, counts in enumerate(method_counts[wind_farm]):
            x_pos = x_base[method_idx] + offsets[wf_idx]
            ax.scatter(np.full(len(counts), x_pos), counts, s=45, color=colors[wf_idx],
                       linewidth=0.4, alpha=0.3, label=wind_farm_names[wf_idx] if method_idx == 0 else "")
            ax.scatter(x_pos, np.median(counts), s=100, color="black", marker="_", zorder=5, linewidth=4)

    ax.set_xticks(x_base)
    ax.set_xticklabels(method_names, rotation=25, ha="right")
    ax.set_ylabel("Number of iterations")
    ax.set_title(f"Number of iterations across AEP methods and wind farms — {wake_model} wake model")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim((0, 800))

    fig.tight_layout()
    plt.savefig(figures_path / f"iterations_all_windfarms_{wake_model.lower()}.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print(f"No iteration data found")


# %%

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
import pandas as pd
import h5py
import numpy as np
from pathlib import Path
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

wind_farms = ["Hornsea_Project_3_HOW03", "Sofia", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea"]
wind_farm_names = ["Hornsea 3", "Sofia", "Thor", "Hornsea 2"]
wake_model = "Gaussian"
figures_path = Path(__file__).parent.parent / "figures"
figures_path.mkdir(exist_ok=True)

aep_models = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "Uniform_CT", "BQ", "RQ"]
method_labels = {
    "360_WD": "360 WD",
    "72_WD": "72 WD",
    "Average_WS": "Average WS",
    "FLOWERS": "FLOWERS",
    "Uniform_CT": "Uniform C$_T$",
    "BQ": "BQ",
    "RQ": "RQ",
}

# ---- Styling ---------------------------------------------------------
# Muted, colorblind-safe palette (Okabe-Ito derived) instead of seaborn defaults
colors = ["#3B6FA0", "#D97A29", "#4F9D69", "#B84C4C"]
edge_colors = ["#274A6B", "#9C5A1C", "#357049", "#853636"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#4A4A4A",
    "axes.labelcolor": "#2B2B2B",
    "text.color": "#2B2B2B",
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
})

n_starts = 30
method_counts = {wf: [] for wf in wind_farms}
method_names = []

# ---- Load data ---------------------------------------------------------
for wind_farm in wind_farms:
    results_file = (
        Path(__file__).parent.parent / "results" / "optimization"
        / "Iteration_3_4WF_30starts" / f"windfarm_{wind_farm}.h5"
    )

    with h5py.File(results_file, "r") as f:
        for aep_model in aep_models:
            counts = []
            for start_id in range(n_starts):
                group_key = f"{aep_model}/{wake_model}/start_{start_id}"
                if group_key not in f:
                    continue
                grp = f[group_key]

                iter_value = None
                for attr_name in ["n_iterations", "iterations", "n_iter", "iter", "n_iter_final", "n_iteration"]:
                    if attr_name in grp.attrs:
                        value = grp.attrs[attr_name]
                        iter_value = int(value) if np.isscalar(value) else int(np.asarray(value).ravel()[0])
                        break

                if iter_value is None:
                    for ds_name in ["n_iterations", "iterations", "n_iter", "iter"]:
                        if ds_name in grp:
                            ds = grp[ds_name]
                            iter_value = int(ds[()]) if ds.shape == () else int(np.asarray(ds).ravel()[0])
                            break

                if iter_value is not None:
                    counts.append(iter_value)

            if counts and wind_farm == wind_farms[0]:
                method_names.append(method_labels.get(aep_model, aep_model))

            if counts:
                method_counts[wind_farm].append(counts)


def beeswarm_jitter(values, width=0.0, n_bins=40, y_max=800):
    """Density-scaled horizontal jitter: points in crowded y-regions spread
    wider, sparse points stay close to center — a lightweight beeswarm."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    bins = np.linspace(0, y_max, n_bins)
    bin_idx = np.clip(np.digitize(values, bins) - 1, 0, n_bins - 2)
    counts_per_bin = np.bincount(bin_idx, minlength=n_bins - 1)
    max_count = counts_per_bin.max() if counts_per_bin.max() > 0 else 1
    local_density = counts_per_bin[bin_idx] / max_count
    rng = np.random.default_rng(42)
    spread = width * (0.25 + 0.75 * local_density)
    return rng.uniform(-1, 1, size=len(values)) * spread


if method_names:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)

    x_base = np.arange(len(method_names))
    group_width = 0.30
    offsets = np.linspace(-group_width, group_width, len(wind_farms))

    # Alternating background bands to help track method columns without
    # leaning on gridlines
    for i in range(len(method_names)):
        if i % 2 == 0:
            ax.add_patch(Rectangle((x_base[i] - 0.5, 0), 1, 800,
                                    facecolor="#F2F2F0", edgecolor="none", zorder=0))

    for wf_idx, wind_farm in enumerate(wind_farms):
        for method_idx, counts in enumerate(method_counts[wind_farm]):
            if not counts:
                continue
            counts_arr = np.asarray(counts)
            jitter = beeswarm_jitter(counts_arr)
            x_pos = x_base[method_idx] + offsets[wf_idx] + jitter

            ax.scatter(
                x_pos, counts_arr,
                s=34, color=colors[wf_idx], edgecolor=edge_colors[wf_idx],
                linewidth=0.35, alpha=0.55, zorder=3,
                label=wind_farm_names[wf_idx] if method_idx == 0 else "",
            )

            median_val = np.median(counts_arr)
            ax.scatter(
                x_base[method_idx] + offsets[wf_idx], median_val,
                s=70, marker="_", color="black", linewidth=3, zorder=5,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(method_names, rotation=20, ha="center", fontsize=12)
    ax.set_ylabel("Number of iterations", fontsize=12, labelpad=8)
    ax.set_title(
        f"{wake_model} wake model",
        fontsize=15, fontweight="medium", pad=14, loc="Center",
    )

    ax.set_xlim(-0.6, len(method_names) - 0.4)
    ax.set_ylim(0, 800)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=10.5)

    legend = ax.legend(
        loc="upper left",
        fontsize=10.5, handletextpad=0.6, borderaxespad=0,
        title="Wind farm", title_fontsize=11,
    )
    legend.get_title().set_fontweight("medium")
    for handle in legend.legend_handles:
        handle.set_alpha(0.9)
        handle.set_sizes([70])

    fig.tight_layout()
    plt.savefig(
        figures_path / f"iterations_all_windfarms_{wake_model.lower()}_styled.png",
        dpi=300, bbox_inches="tight",
    )
    plt.show()

    # Print table with median and std for each farm and AEP model
    print("\n" + "="*80)
    print("Median and Standard Deviation of Iterations by Wind Farm and AEP Model")
    print("="*80)
    
    # Create a detailed table
    table_data = []
    for wf_idx, wind_farm in enumerate(wind_farms):
        for method_idx, counts in enumerate(method_counts[wind_farm]):
            if counts:
                counts_arr = np.asarray(counts)
                median_val = np.median(counts_arr)
                std_val = np.std(counts_arr)
                table_data.append({
                    'Wind Farm': wind_farm_names[wf_idx],
                    'AEP Model': method_names[method_idx],
                    'Median': f"{median_val:.1f}",
                    'Std Dev': f"{std_val:.2f}",
                    'Count': len(counts)
                })
    
    if table_data:
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        df.to_csv(figures_path / f"iterations_summary_{wake_model.lower()}.csv", index=False)
        print("="*80)
else:
    print("No iteration data found")


# %% --- Number of multistarts to optimize BQ sampling points ---

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

# Load data
path_file = Path(__file__).parent.parent / "results" / "BQ_points_results.csv"
AEPs = pd.read_csv(path_file)
N_attempts = np.arange(1, 51, 2)
N_runs = 20
aep_ref = 719.61

mean_aep = np.array(list(AEPs.mean(axis=0)))
std_aep = np.array(list(AEPs.std(axis=0)))

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax = axes[0]
ax.plot(N_attempts, mean_aep, marker='o', label='Mean AEP for 20 runs')
ax.fill_between(N_attempts,
                mean_aep - std_aep,
                mean_aep + std_aep,
                alpha=0.3, label='Standard Deviation')
ax.axhline(aep_ref, color='r', linestyle='--', label='Reference AEP')
ax.set_ylabel('AEP [GWh]', fontsize=14)
ax.set_title(f'AEP and Standard Deviation vs Number of Multistarts ({N_runs} runs)', fontsize=16)
ax.legend(fontsize=14)
ax.tick_params(labelsize=14)

ax = axes[1]
ax.plot(N_attempts, std_aep, marker='o', color='orange', label='Standard deviation of AEP for 20 runs')
# Exponential decay to a floor: std ~ a * exp(-b * N) + c
from scipy.optimize import curve_fit
def exp_floor(N, a, b, c):
    return a * np.exp(-b * N) + c
try:
    p0 = [std_aep[0] - std_aep[-1], 0.1, std_aep[-1]]
    popt, _ = curve_fit(exp_floor, N_attempts, std_aep, p0=p0, maxfev=5000)
    trend = exp_floor(N_attempts, *popt)
    ax.plot(N_attempts, trend, '--', color='gray',
            label=f'Exponential fit')
except RuntimeError:
    pass  # fit failed, skip trend line
ax.set_xlabel('Number of Multistarts', fontsize=14)
ax.set_ylabel('Std of AEP [GWh]', fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig('BQ_points_convergence.pdf', dpi=300, bbox_inches='tight')
plt.show()