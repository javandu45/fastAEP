from FAST_AEP.optimization import optifast
from FAST_AEP.utils import generic_site, turbine_generator
from FAST_AEP.BQ import bayesian_quadrature
from FAST_AEP.FLOWERS import NOJ_flowers, TurbOPark_flowers, gaussian_flowers
from FAST_AEP.basic_wfm import WD_Bins, average_WS, uniform_CT

from py_wake.literature.noj import Jensen_1983
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils.gradients import autograd

from FAST_AEP.utils import get_wind_farm_data, generate_random_array, get_limits

import time
import matplotlib.pyplot as plt
import numpy as np

wind_farm = "Sofia"

site = generic_site(wind_farm)
turbine = turbine_generator(wind_farm)

wind_farm_data = get_wind_farm_data(wind_farm)

# Print wind farm data
print(f"Wind Farm: {wind_farm}")
print(f"Number of Turbines: {wind_farm_data['turbine_count'].values[0]}")
print(f"Capacity: {wind_farm_data['capacity_mw'].values[0]} MW")

def setup_BQ_wfm(deficit_model, aep_method):

    if deficit_model == "NOJ":
        wfm = Jensen_1983(site, turbine, k=0.05)
    elif deficit_model == "Gaussian":
        wfm = Bastankhah_PorteAgel_2014(site, turbine, k=0.05)
    elif deficit_model == "TurbOPark":
        deficit_model = TurboNOJDeficit()
        wfm = PropagateDownwind(site, turbine,
                                wake_deficitModel=deficit_model,
                                superpositionModel=SquaredSum(),
                                rotorAvgModel=AreaOverlapAvgModel())

    limits = get_limits(wind_farm)

    x, y = generate_random_array(n_tur=wind_farm_data['turbine_count'].values[0], turbine=turbine, spacing=2, limits=limits)

    BQ_wfm = bayesian_quadrature(site=site,
                                 windTurbines=turbine,
                                 flow_model=wfm,
                                 x0=x,
                                 y0=y,
                                 N_train=3000,
                                 N_MC = 4000,
                                 aep_method=aep_method)
    
    BQ_wfm.train_and_get_kernel()
    BQ_wfm.optimize_BQ_points(N_points=360, N_attempts=5, jitter=0.1)
    BQ_wfm.setup_gradients(gradient_method=autograd, n_cpu=8)

    BQ_wfm.plot_optimized_points()
    plt.savefig("BQ_points.png", dpi=300)

    return BQ_wfm

# RQ_NOJ = setup_BQ_wfm("NOJ", "RQ")

evaluating_wfm = WD_Bins(site=site, windTurbines=turbine, deficit_model="TurbOPark", n_bins=360)

wfm_flowers = TurbOPark_flowers(site=site, windTurbines=turbine, n_terms=10)

# wfm_flowers = WD_Bins(site=site, windTurbines=turbine, deficit_model="TurbOPark", n_bins=360)

# wfm_flowers = setup_BQ_wfm("Gaussian", "BQ")

# ########################################################################################
# First optimization - Not including distance constraints
# optimization_problem = optifast(wind_farm=wind_farm, wind_farm_model=wfm_flowers, min_spacing=None, n_cpu=8)

# tf_problem = optimization_problem.setup_problem(tolerance=1e-6, expected_cost=10, max_iter=100)

# print("\n" + "-" * 50)
# print("Starting optimization...")
# time_start = time.time()
# _, state, recorder = tf_problem.optimize()
# time_end = time.time()
# time_1 = time_end - time_start

# if optimization_problem.normalization:
#     x_opt = state["x_norm"] * optimization_problem.max_x
#     y_opt = state["y_norm"] * optimization_problem.max_y

# else:
#     x_opt = state["x"]
#     y_opt = state["y"]

# limits = optimization_problem.wf_limits

# # Compute distance between turbines
# dist = []
# for i in range(len(x_opt)):
#     for j in range(i + 1, len(x_opt)):
#         dist.append(np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2))

# D = optimization_problem.windTurbines.diameter()

# print(f"Optimization completed in {time_1:.2f} seconds")
# print(f"Optimized AEP: {evaluating_wfm.aep(x_opt, y_opt):.2f} GWh")
# print(f"Minimum distance as multiple of turbine diameter: {np.min(dist)/D:.2f}")

# convergence_1 = recorder.get("cost")

# # Plot convergence
# # plt.figure(figsize=(8, 5))
# # plt.plot(-convergence, marker='o')
# # plt.title("AEP Optimization Convergence")
# # plt.xlabel("Iteration")
# # plt.ylabel("AEP (GWh)")
# # plt.grid()
# # plt.show()

# ########################################################################################
# # Second optimization - Including distance constraints
# optimization_problem = optifast(wind_farm=wind_farm, wind_farm_model=wfm_flowers, min_spacing=4, x_0=x_opt, y_0=y_opt, n_cpu=8)

# tf_problem = optimization_problem.setup_problem(tolerance=1e-3, expected_cost=10, max_iter=100)

# print("\n")
# print("-" * 50)
# print("Starting optimization with distance constraints...")
# time_start = time.time()
# _, state, recorder = tf_problem.optimize()
# time_end = time.time()
# time_2 = time_end - time_start

# if optimization_problem.normalization:
#     x_opt = state["x_norm"] * optimization_problem.max_x
#     y_opt = state["y_norm"] * optimization_problem.max_y
# else:
#     x_opt = state["x"]
#     y_opt = state["y"]


# limits = optimization_problem.wf_limits

# # Compute distance between turbines
# dist = []
# for i in range(len(x_opt)):
#     for j in range(i + 1, len(x_opt)):
#         dist.append(np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2))

# D = optimization_problem.windTurbines.diameter()

# print(f"Optimization completed in {time_2:.2f} seconds")
# print(f"Optimized AEP: {evaluating_wfm.aep(x_opt, y_opt):.2f} GWh")
# print(f"Minimum distance as multiple of turbine diameter: {np.min(dist)/D:.2f}")

# aep_2_tier = evaluating_wfm.aep(x_opt, y_opt)
# convergence_2 = recorder.get("cost")

########################################################################################
# Total optimization - Including distance constraints from the beginning
optimization_problem = optifast(wind_farm=wind_farm, wind_farm_model=wfm_flowers, min_spacing=3, n_cpu=8)

tf_problem = optimization_problem.setup_problem(tolerance=1e-6, expected_cost=10, max_iter=100)

print("\n")
print("-" * 50)
print("Starting optimization with distance constraints...")
time_start = time.time()
_, state, recorder = tf_problem.optimize()
time_end = time.time()
time_3 = time_end - time_start

if optimization_problem.normalization:
    x_opt = state["x_norm"] * optimization_problem.max_x
    y_opt = state["y_norm"] * optimization_problem.max_y
else:
    x_opt = state["x"]
    y_opt = state["y"]


limits = optimization_problem.wf_limits

# Compute distance between turbines
dist = []
for i in range(len(x_opt)):
    for j in range(i + 1, len(x_opt)):
        dist.append(np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2))

D = optimization_problem.windTurbines.diameter()

print(f"Optimization completed in {time_3:.2f} seconds")
print(f"Optimized AEP: {evaluating_wfm.aep(x_opt, y_opt):.2f} GWh")
print(f"Minimum distance as multiple of turbine diameter: {np.min(dist)/D:.2f}")

# aep_1_tier = evaluating_wfm.aep(x_opt, y_opt)
convergence_3 = recorder.get("cost")

# print("\n" + "#" * 50)
# print(f"Total two-tier optimization time: {time_1 + time_2:.2f} seconds")
# print(f"Total one-tier optimization time: {time_3:.2f} seconds")
# print(f"Two-tier optimization AEP: {aep_2_tier:.2f} GWh")
# print(f"One-tier optimization AEP: {aep_1_tier:.2f} GWh")
# print("#" * 50)


# Plot all convergence curves
fig, axes = plt.subplots(1, 1, figsize=(6, 5))

# axes[0].plot(-convergence_1, marker='o')
# axes[0].set_title('Two-tier Optimization\n(No Distance Constraints)')
# axes[0].set_xlabel('Iteration')
# axes[0].set_ylabel('Negative AEP')
# axes[0].grid()

# axes[1].plot(-convergence_2, marker='o')
# axes[1].set_title('Two-tier Optimization\n(With Distance Constraints)')
# axes[1].set_xlabel('Iteration')
# axes[1].grid()

axes.plot(-convergence_3, marker='o')
axes.set_title('One-tier Optimization\n(With Distance Constraints)')
axes.set_xlabel('Iteration')
axes.grid()

fig.suptitle("AEP Optimization Convergence")
plt.tight_layout()
plt.show()
plt.savefig("optimization_convergence.png", dpi=300)

# Plot the optimized layout
plt.figure(figsize=(10, 10))
plt.scatter(x_opt, y_opt, c='blue', label='Optimized Turbine Positions')
plt.plot(limits[:, 0], limits[:, 1], 'k-', label='Wind Farm Boundary')
plt.title(f"Optimized Wind Farm Layout for {wind_farm}")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.axis('equal')
plt.legend()
plt.grid()
plt.savefig("optimized_layout.png", dpi=300)
plt.show()

