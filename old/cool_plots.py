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
# %% # PLOT AEP BENCHMARKING

results_path = str(Path(__file__).parent.parent / "results" / "benchmarking")
# results_path = "results/benchmarking"

# #############
# AEP_VALUES
# Read results
aeps_noj = pd.read_csv(results_path + "/benchmark_aep_noj.csv")
aeps_gaussian = pd.read_csv(results_path + "/benchmark_aep_gaussian.csv")
aeps_turbopark = pd.read_csv(results_path + "/benchmark_aep_turbopark.csv")

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
# %% PLOT AEP AND TIME BENCHMARKING IN A SINGLE PLOT


results_path = str(Path(__file__).parent.parent / "results" / "benchmarking")
# results_path = "results/benchmarking"

use_parallelization = False

# #############
# AEP_VALUES
# Read results
aeps_noj = pd.read_csv(results_path + "/benchmark_aep_noj.csv")
aeps_gaussian = pd.read_csv(results_path + "/benchmark_aep_gaussian.csv")
aeps_turbopark = pd.read_csv(results_path + "/benchmark_aep_turbopark.csv")

# #############
# TIMES VALUES
if use_parallelization:
    times_noj = pd.read_csv(results_path + "/benchmark_runtime_noj.csv")
    times_gaussian = pd.read_csv(results_path + "/benchmark_runtime_gaussian.csv")
    times_turbopark = pd.read_csv(results_path + "/benchmark_runtime_turbopark.csv")
else:
    times_noj = pd.read_csv(results_path + "/benchmark_runtime_noj_no_parallel.csv")
    times_gaussian = pd.read_csv(results_path + "/benchmark_runtime_gaussian_no_parallel.csv")
    times_turbopark = pd.read_csv(results_path + "/benchmark_runtime_turbopark_no_parallel.csv")

turbine_ranges = [25, 50, 75, 100, 150, 200, 250, 300]
aep_methods = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "BQ", "RQ", "Uniform_CT"]

def plot_benchmark(ax, data, title, ylabel, xlabel):
    for aep_method in aep_methods:
        y_values = 100 * (np.array(data[aep_method]) - np.array(data["360_WD"])) / np.array(data["360_WD"])
        ax.plot(turbine_ranges, y_values, label=aep_method, marker="o", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()

fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=300, sharey="row", sharex=True)
axs = axs.ravel()
for i, (data, title, ylabel, xlabel) in enumerate(zip(
    [aeps_noj, aeps_gaussian, aeps_turbopark, times_noj, times_gaussian, times_turbopark],
    ["NOJ", "Gaussian", "TurbOPark", "", "", ""],
    ["AEP difference relative to 360 WD [%]", "", "", "Time difference relative to 360 WD [s]", "", ""],
    ["", "", "", "Number of turbines", "Number of turbines", "Number of turbines"]
)):
    plot_benchmark(axs[i], data, title, ylabel, xlabel)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(aep_methods), bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()

######################################################################
# %% AEP scatter plots for different methods and wake models

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

wake_model = "TurbOPark"  # "NOJ", "Gaussian", "TurbOPark"
aep_models = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "Uniform_CT", "BQ", "RQ", "SGD"]

wind_farms = ["Hornsea_Project_3_HOW03", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea", "Sofia"]
wind_farm_names = ["Hornsea 3 \n (231 WT)", "Thor \n (72 WT)", "Hornsea 2 \n (55 WT)", "Sofia \n (100 WT)"]

results_dir = Path("results/optimization")
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
                "AEP_GWh_results": aeps,
                "time_s_results": times,
            })
    tables[wf] = pd.DataFrame(rows).set_index("AEP_model")

# %% ----- Plot wind farm boundaries -----

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


for ax, wf, wf_name in zip(axes, wind_farms, wind_farm_names):

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

plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.savefig("wind_farm_boundaries.png", dpi=300, bbox_inches="tight")
plt.show()

# %% ----- Wind farms wind roses -----

import matplotlib.pyplot as plt
import numpy as np
from FAST_AEP.utils import generic_site
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

wind_farms = ["Hornsea_Project_3_HOW03", "Sofia", "Thor", "Hornsea_Project_2_-_Phase_1_Breesea"]
wind_farm_names = ["Hornsea 3 \n (231 WT, C)", "Sofia \n (100 WT, C, DC)", "Thor \n (72 WT, NC)", "Hornsea 2 \n (55 WT, NC, DC)"]

# Define wind speed bins (m/s)
speed_bins = np.array([0, 5, 10, 15, 20, 25])
colors = ['#0570B0', '#FF7F00', '#2CA02C', '#D62728', '#9467BD']  # blue, orange, green, red, purple
n_speed_bins = len(colors)

# Create figure with 2x2 subplots for all 4 wind farms
fig = plt.figure(figsize=(8, 8))

for idx, (wf, wf_name) in enumerate(zip(wind_farms, wind_farm_names)):

    site = generic_site(wind_farm=wf)
    p = site.ds.Sector_frequency.values      # shape (360,)
    A = site.ds.Weibull_A.values             # shape (360,)
    k = site.ds.Weibull_k.values             # shape (360,)

    # Bin 360 sectors into 12 sectors (30° each)
    n_sectors = 12
    sector_size = 360 // n_sectors
    
    # Frequency in each speed bin for each direction sector
    freq_by_speed = np.zeros((n_sectors, n_speed_bins))
    
    for i in range(n_sectors):
        idx_start = i * sector_size
        idx_end = (i + 1) * sector_size
        
        # Get mean Weibull parameters for this direction sector
        A_sector = np.mean(A[idx_start:idx_end])
        k_sector = np.mean(k[idx_start:idx_end])
        p_sector = np.mean(p[idx_start:idx_end])
        
        # Compute CDF for Weibull distribution
        # P(v < speed) = 1 - exp(-(speed/A)^k)
        cdf = 1 - np.exp(-(speed_bins / A_sector) ** k_sector)
        
        # Frequency for each speed bin = p_sector * (CDF[i+1] - CDF[i])
        freq_by_speed[i, :] = p_sector * np.diff(cdf)
    
    # Wind directions for the 12 sectors
    directions = np.arange(n_sectors) * sector_size + sector_size / 2
    theta = np.deg2rad(directions)
    
    # Create polar subplot
    ax = fig.add_subplot(2, 2, idx + 1, projection='polar')
    
    # Stacked bars: bottom of each bar starts at 0, then we stack the speed bins
    bottom = np.zeros(n_sectors)
    for speed_idx in range(n_speed_bins):
        ax.bar(theta, freq_by_speed[:, speed_idx], width=np.deg2rad(sector_size),
               bottom=bottom, label=f'{speed_bins[speed_idx]:.1f}-{speed_bins[speed_idx+1]:.1f} m/s',
               color=colors[speed_idx], edgecolor='black', linewidth=0.5, alpha=0.8)
        bottom += freq_by_speed[:, speed_idx]
    
    # Labels and formatting
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_title(f'Wind Rose: {wf_name}', pad=20, fontsize=12, weight='bold')

    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f'{int(d)}°' for d in np.arange(0, 360, 30)], fontsize=12)
    ax.set_yticklabels([])

# Create a single legend in the center for all subplots
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5), frameon=True, fontsize=12)

plt.tight_layout()
plt.show()


# %%

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

fig, axes = plt.subplots(2, 2, figsize=(7, 8.5), subplot_kw={'projection': 'polar'})
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

    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f'{int(d)}°' for d in np.arange(0, 360, 30)], fontsize=8.5, color='#444444')

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
plt.show()