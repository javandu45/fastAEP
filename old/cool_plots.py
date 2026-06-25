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
from FAST_AEP.utils import generic_site

D = V80().diameter()
fontsize = 14

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
#     baseline_aep = proxy.new_aep(base_layout[:, 0], base_layout[:, 1])

#     # Get turbine diameter and calculate minimum spacing (2D)
#     turbine_diameter = proxy.windTurbines.diameter()
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
#             aep_values[i, j] = proxy.new_aep(x_all, y_all) - baseline_aep

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

# site = generic_site(ws=10)

# wfm_pywake = Jensen_1983(site=site, windTurbines=V80(), k=0.04)
# wfm_pywake_gaussian = Bastankhah_PorteAgel_2014(site=site, windTurbines=V80(), k=0.04)
# wfm_pywake_TurbOPark = Nygaard_2022(site=site, windTurbines=V80())
# x, y = Hornsrev1Site().initial_position.T

# wfm_NOJ = NOJ_flowers(site=site, windTurbines=V80(), n_terms=10)
# wfm_gaussian = gaussian_flowers(site=site, windTurbines=V80(), n_terms=10)
# wfm_TurbOPark = TurbOPark_flowers(site=site, windTurbines=V80(), n_terms=10)


# wfm_BQ_150 = bayesian_quadrature(site=site, windTurbines=V80(), flow_model=wfm_pywake, x0=x, y0=y, N_train=3000)
# wfm_BQ_150.train_and_get_kernel()
# wfm_BQ_150.optimize_BQ_points(N_points=150, N_attempts=10)

# wfm_BQ_100 = bayesian_quadrature(site=site, windTurbines=V80(), flow_model=wfm_pywake, x0=x, y0=y, N_train=3000)
# wfm_BQ_100.train_and_get_kernel()
# wfm_BQ_100.optimize_BQ_points(N_points=100, N_attempts=10)

# wfm_BQ_75 = bayesian_quadrature(site=site, windTurbines=V80(), flow_model=wfm_pywake, x0=x, y0=y, N_train=3000)
# wfm_BQ_75.train_and_get_kernel()
# wfm_BQ_75.optimize_BQ_points(N_points=75, N_attempts=10)

# wfm_BQ_50 = bayesian_quadrature(site=site, windTurbines=V80(), flow_model=wfm_pywake, x0=x, y0=y, N_train=3000)
# wfm_BQ_50.train_and_get_kernel()
# wfm_BQ_50.optimize_BQ_points(N_points=50, N_attempts=10)

# wfm_BQ_30 = bayesian_quadrature(site=site, windTurbines=V80(), flow_model=wfm_pywake, x0=x, y0=y, N_train=3000)
# wfm_BQ_30.train_and_get_kernel()
# wfm_BQ_30.optimize_BQ_points(N_points=30, N_attempts=10)

# # wfm_flowers = NOJ_flowers(site=site, windTurbines=V80(), n_terms=10)

# # proxies = [wfm_pywake, wfm_BQ, wfm_flowers]
# # names = ["PyWake", "Bayesian Quadrature", "FLOWERS"]

# proxies = [wfm_BQ_150, wfm_BQ_100, wfm_BQ_75, wfm_BQ_50, wfm_BQ_30]
# names = ["Bayesian Quadrature (150 points)", "Bayesian Quadrature (100 points)", "Bayesian Quadrature (75 points)", "Bayesian Quadrature (50 points)", "Bayesian Quadrature (30 points)"]
# # proxies = [wfm_pywake_NOJ, wfm_NOJ, wfm_pywake_gaussian, wfm_gaussian, wfm_pywake_TurbOPark, wfm_TurbOPark]
# # names = ["PyWake NOJ", "FLOWERS NOJ", "PyWake Gaussian", "FLOWERS Gaussian", "PyWake TurbOPark", "FLOWERS TurbOPark"]


# # %%

# # Create figure with 6 subplots
# fig, axes = plt.subplots(3, 2, figsize=(13, 18), dpi=200)
# axes = axes.ravel()

# contourf = None
# for i, proxy in enumerate(proxies):
#     # Plot AEP impact for each proxy and get contourf object
#     contourf = plot_9th_turbine_aep_impact(proxy, axes[i], grid_resolution=100)
#     axes[i].set_title(f"Proxy: {names[i]}", fontsize=fontsize+2)

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

# plt.show()

# %%
# PLOT AEP BENCHMARKING

# results_path = Path(__file__).parent.parent / "results" / "benchmarking"
results_path = "results/benchmarking"

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
plt.savefig(results_path + "/benchmark_aep_noj.png", dpi=300)
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
plt.savefig(results_path + "/benchmark_aep_gaussian.png", dpi=300)
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
plt.savefig(results_path + "/benchmark_aep_turbopark.png", dpi=300)
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
plt.savefig(results_path + "/benchmark_runtime_noj.png", dpi=300)
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
plt.savefig(results_path + "/benchmark_runtime_gaussian.png", dpi=300)
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
plt.savefig(results_path + "/benchmark_runtime_turbopark.png", dpi=300)
plt.show()

