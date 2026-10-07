
# A code to run a single optimization with the input data in the code

from FAST_AEP.optimization import optifast
from FAST_AEP.utils import generic_site, turbine_generator, build_aep_model
from FAST_AEP.BQ import bayesian_quadrature
from FAST_AEP.FLOWERS import NOJ_flowers, TurbOPark_flowers, gaussian_flowers
from FAST_AEP.basic_wfm import WD_Bins, average_WS, uniform_CT
from FAST_AEP.SGD import SGD

from py_wake.literature.noj import Jensen_1983
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils.gradients import autograd

from FAST_AEP.utils import get_wind_farm_data, generate_random_array, get_limits

import time
import matplotlib.pyplot as plt
import numpy as np

def main():

    wind_farm = "Sofia"
    wake_model = "NOJ"
    aep_method = "BQ"
    n_cpu = 1
    tol = 1e-6

    # Set up optimization
    # Excpected costs
    if wake_model == "NOJ":
        expected_cost = 10
    elif wake_model == "Gaussian":
        expected_cost = 1
        if aep_method == "FLOWERS":
            expected_cost = 1e-4
    elif wake_model == "TurbOPark":
        expected_cost = 10
        if aep_method == "360_WD":
            expected_cost = 1
            tol = 1e-6

    # Maximum number of iterations
    max_iter = 150
    if (aep_method == "RQ" or aep_method == "BQ") and wake_model == "Gaussian":
        max_iter = 100
    elif aep_method == "SGD":
        max_iter = 2000

    # Distancing constraints
    min_spacing = None  # Default is no distancing constraints
    if wind_farm in ["Hornsea_Project_2_-_Phase_1_Breesea", "Sofia"]:
        min_spacing = 3

    # Normalization
    normalization = True
    if aep_method == "SGD":
        normalization = False

    site = generic_site(wind_farm)
    turbine = turbine_generator(wind_farm)

    wind_farm_data = get_wind_farm_data(wind_farm)

    # Print wind farm data
    print(f"Wind Farm: {wind_farm}")
    print(f"Number of Turbines: {wind_farm_data['turbine_count'].values[0]}")
    print(f"Capacity: {wind_farm_data['capacity_mw'].values[0]} MW")

    evaluating_wfm = WD_Bins(site=site, windTurbines=turbine, deficit_model=wake_model, n_bins=360)

    wfm = build_aep_model(wind_farm=wind_farm, deficit_model=wake_model, aep_method=aep_method, k=0.05, n_cpu=n_cpu)

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
    optimization_problem = optifast(wind_farm=wind_farm, wind_farm_model=wfm, min_spacing=min_spacing, n_cpu=n_cpu, normalization=True, seed=1)

    tf_problem = optimization_problem.setup_problem(tolerance=tol, expected_cost=expected_cost, max_iter=200)

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
    # plt.savefig("optimization_convergence.png", dpi=300)

    # Plot the optimized layout
    plt.figure(figsize=(5, 5))
    plt.scatter(x_opt, y_opt, c='blue', label='Optimized Turbine Positions')
    plt.plot(limits[:, 0], limits[:, 1], 'k-', label='Wind Farm Boundary')
    plt.title(f"Optimized Wind Farm Layout for {wind_farm}")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid()
    # plt.savefig("optimized_layout.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()