
# Benchamrking different AEP methods with different wake models and different number of turbines.
# Both computation time and AEP.

from FAST_AEP.utils import generic_site, generate_array, turbine_generator, basic_wfm, build_wfm
from FAST_AEP import FLOWERS
from FAST_AEP.BQ import bayesian_quadrature
import time

print("Imported all necessary modules")

aep_methods = ["360_WD", "72_WD", "Average_WS", "FLOWERS", "RQ", "BQ", "Uniform_CT"]
wake_models = ["NOJ", "Gaussian", "TurbOPark"]

turbine_ranges = [25, 50, 75, 100, 150, 200, 250, 300]

site = generic_site("Sofia")
turbine = turbine_generator("Sofia")

k = 0.05

def build_aep_model(aep_method, deficit_model, n_turbines):

    wfm = build_wfm(site, turbine, deficit_model, k)

    if aep_method == "360_WD":
        aep_model = basic_wfm.WD_Bins(site=site, windTurbines=turbine, deficit_model=deficit_model, n_bins=360, k=k)
    elif aep_method == "72_WD":
        aep_model = basic_wfm.WD_Bins(site=site, windTurbines=turbine, deficit_model=deficit_model, n_bins=72, k=k)
    elif aep_method == "Average_WS":
        aep_model = basic_wfm.average_WS(site=site, windTurbines=turbine, deficit_model=deficit_model, k=k)
    elif aep_method == "Uniform_CT":
        aep_model = basic_wfm.uniform_CT(site=site, windTurbines=turbine, deficit_model=deficit_model, k=k)
    elif aep_method == "FLOWERS":
        if deficit_model == "NOJ":
            aep_model = FLOWERS.NOJ_flowers(site=site, windTurbines=turbine, n_terms=10, k=k)
        elif deficit_model == "Gaussian":
            aep_model = FLOWERS.gaussian_flowers(site=site, windTurbines=turbine, n_terms=10, k=k)
        elif deficit_model == "TurbOPark":
            aep_model = FLOWERS.TurbOPark_flowers(site=site, windTurbines=turbine, n_terms=10)
    elif aep_method == "BQ":
        x, y = generate_array(n_tur=n_turbines, turbine=turbine, spacing=5)
        aep_model = bayesian_quadrature(site=site,
                                       windTurbines=turbine,
                                       flow_model=wfm,
                                       x0=x,
                                       y0=y,
                                       N_train=3000,
                                       N_MC = 4000,
                                       aep_method=aep_method)
        aep_model.train_and_get_kernel()
        aep_model.optimize_BQ_points(N_points=360, N_attempts=10)
    elif aep_method == "RQ":
        x, y = generate_array(n_tur=n_turbines, turbine=turbine, spacing=5)
        aep_model = bayesian_quadrature(site=site,
                                       windTurbines=turbine,
                                       flow_model=wfm,
                                       x0=x,
                                       y0=y,
                                       N_train=3000,
                                       N_MC = 4000,
                                       aep_method=aep_method)
        aep_model.train_and_get_kernel()
        aep_model.optimize_BQ_points(N_points=360, N_attempts=10)
    return aep_model


# ########################################################################################
# NOJ

aeps_noj = {}
times_noj = {}

for aep_method in aep_methods:

    aeps_noj[aep_method] = []
    times_noj[aep_method] = []

    print(f"Running {aep_method} with NOJ wake model")
    wfm = build_aep_model(aep_method=aep_method, deficit_model="NOJ", n_turbines=100)

    for n_turbines in turbine_ranges:
        aeps_runs = []
        times_runs = []

        for run in range(1):
            x, y = generate_array(n_tur=n_turbines, turbine=turbine, spacing=5)

            if aep_method == "FLOWERS":
                time_start = time.time()
                aep = wfm.aep(x, y)
                time_end = time.time()
            else:
                time_start = time.time()
                aep = wfm.aep(x, y)
                time_end = time.time()

            aeps_runs.append(aep)
            times_runs.append(time_end - time_start)

        aeps_noj[aep_method].append(sum(aeps_runs) / len(aeps_runs))
        times_noj[aep_method].append(sum(times_runs) / len(times_runs))


# ########################################################################################
# Gaussian

aeps_gaussian = {}
times_gaussian = {}

for aep_method in aep_methods:

    aeps_gaussian[aep_method] = []
    times_gaussian[aep_method] = []

    print(f"Running {aep_method} with Gaussian wake model")
    wfm = build_aep_model(aep_method=aep_method, deficit_model="Gaussian", n_turbines=100)

    for n_turbines in turbine_ranges:
        aeps_runs = []
        times_runs = []

        for run in range(1):
            x, y = generate_array(n_tur=n_turbines, turbine=turbine, spacing=5)

            if aep_method == "FLOWERS":
                time_start = time.time()
                aep = wfm.aep(x, y)
                time_end = time.time()
            else:
                time_start = time.time()
                aep = wfm.aep(x, y)
                time_end = time.time()

            aeps_runs.append(aep)
            times_runs.append(time_end - time_start)

        aeps_gaussian[aep_method].append(sum(aeps_runs) / len(aeps_runs))
        times_gaussian[aep_method].append(sum(times_runs) / len(times_runs))

    
# ########################################################################################
# TurbOPark

aeps_turbopark = {}
times_turbopark = {}

for aep_method in aep_methods:

    aeps_turbopark[aep_method] = []
    times_turbopark[aep_method] = []

    print(f"Running {aep_method} with TurbOPark wake model")
    wfm = build_aep_model(aep_method=aep_method, deficit_model="TurbOPark", n_turbines=100)

    for n_turbines in turbine_ranges:
        aeps_runs = []
        times_runs = []

        for run in range(1):
            x, y = generate_array(n_tur=n_turbines, turbine=turbine, spacing=5)

            if aep_method == "FLOWERS":
                time_start = time.time()
                aep = wfm.aep(x, y)
                time_end = time.time()
            else:
                time_start = time.time()
                aep = wfm.aep(x, y)
                time_end = time.time()

            aeps_runs.append(aep)
            times_runs.append(time_end - time_start)

        aeps_turbopark[aep_method].append(sum(aeps_runs) / len(aeps_runs))
        times_turbopark[aep_method].append(sum(times_runs) / len(times_runs))

# Save_results in csv files
import pandas as pd

df_noj = pd.DataFrame(aeps_noj, index=turbine_ranges)
df_noj.to_csv("results/benchmark_aep_noj_2.csv")
df_gaussian = pd.DataFrame(aeps_gaussian, index=turbine_ranges)
df_gaussian.to_csv("results/benchmark_aep_gaussian_2.csv")
df_turbopark = pd.DataFrame(aeps_turbopark, index=turbine_ranges)
df_turbopark.to_csv("results/benchmark_aep_turbopark_2.csv")

# df_noj_time = pd.DataFrame(times_noj, index=turbine_ranges)
# df_noj_time.to_csv("results/benchmark_runtime_noj_no_parallel.csv")
# df_gaussian_time = pd.DataFrame(times_gaussian, index=turbine_ranges)
# df_gaussian_time.to_csv("results/benchmark_runtime_gaussian_no_parallel.csv")
# df_turbopark_time = pd.DataFrame(times_turbopark, index=turbine_ranges)
# df_turbopark_time.to_csv("results/benchmark_runtime_turbopark_no_parallel.csv")


# Plot results
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    plt.plot(turbine_ranges, np.array(aeps_noj[aep_method]) / np.array(aeps_noj["360_WD"]), label=aep_method)
plt.xlabel("Number of turbines")
plt.ylabel("AEP [GWh]")
plt.title("AEP for different methods with NOJ wake model")
plt.legend()
plt.savefig("results/benchmark_aep_noj_no_parallel.png")

plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    plt.plot(turbine_ranges, np.array(aeps_gaussian[aep_method]) / np.array(aeps_gaussian["360_WD"]), label=aep_method)
plt.xlabel("Number of turbines")
plt.ylabel("AEP [GWh]")
plt.title("AEP for different methods with Gaussian wake model")
plt.legend()
plt.savefig("results/benchmark_aep_gaussian_no_parallel.png")

plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    plt.plot(turbine_ranges, np.array(aeps_turbopark[aep_method]) / np.array(aeps_turbopark["360_WD"]), label=aep_method)
plt.xlabel("Number of turbines")
plt.ylabel("AEP [GWh]")
plt.title("AEP for different methods with TurbOPark wake model")
plt.legend()
plt.savefig("results/benchmark_aep_turbopark_no_parallel.png")

plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    plt.plot(turbine_ranges, times_noj[aep_method], label=aep_method)
plt.xlabel("Number of turbines")
plt.ylabel("Runtime [s]")
plt.title("Runtime for different methods with NOJ wake model")
plt.legend()
plt.savefig("results/benchmark_runtime_noj_no_parallel.png")

plt.figure(figsize=(10, 6))
for aep_method in aep_methods:
    plt.plot(turbine_ranges, times_gaussian[aep_method], label=aep_method)
plt.xlabel("Number of turbines")
plt.ylabel("Runtime [s]")
plt.title("Runtime for different methods with Gaussian wake model")
plt.legend()
plt.savefig("results/benchmark_runtime_gaussian_no_parallel.png")

plt.figure(figsize=(10, 6))
for aep_method in aep_methods: 
    plt.plot(turbine_ranges, times_turbopark[aep_method], label=aep_method)
plt.xlabel("Number of turbines")
plt.ylabel("Runtime [s]")
plt.title("Runtime for different methods with TurbOPark wake model")
plt.legend()
plt.savefig("results/benchmark_runtime_turbopark_no_parallel.png")