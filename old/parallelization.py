
# A code to obtain the computational time for the different AEP methods 
# for an increasing number of CPUs, for the selected wind farm size.

from FAST_AEP.basic_wfm import WD_Bins, average_WS, uniform_CT
from FAST_AEP.FLOWERS import NOJ_flowers, gaussian_flowers, TurbOPark_flowers
from FAST_AEP.BQ import bayesian_quadrature
from FAST_AEP.SGD import SGD

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80

import time
import pandas as pd
import numpy as np
from FAST_AEP.utils import generic_site, build_wfm, generate_random_array
import warnings
warnings.filterwarnings("ignore")

# Wind farm conditions and layout generator
site = generic_site("Dogger_Bank_C")
turbines = V80()
x, y = Hornsrev1Site().initial_position.T  # reference layout for GP training

def _random_square_layout(n_tur: int, spacing: int):
    side = int(np.sqrt(n_tur)) * spacing * turbines.diameter() * 1.5  # add some extra space
    limits = np.array([(0, 0), (side, 0), (side, side), (0, side)])
    return generate_random_array(n_tur=n_tur, turbine=turbines, spacing=spacing, limits=limits)


# Function to set up BQ wind farm models
def setup_BQ_wfm(base_wfm, aep_method):

    BQ_wfm = bayesian_quadrature(site=site,
                                 windTurbines=turbines,
                                 flow_model=base_wfm,
                                 x0=x,
                                 y0=y,
                                 N_train=100,
                                 N_MC = 4000,
                                 aep_method=aep_method)
    
    BQ_wfm.train_and_get_kernel()
    BQ_wfm.optimize_BQ_points(N_points=360, N_attempts=1)

    return BQ_wfm


# Generate testing layout
x, y = _random_square_layout(50, spacing=4)  # 100 turbines with 4 rotor diameters spacing

# Function to set up wind farm models
# 1. 360 WD
# 2. 72 WD
# 3. Average WS
# 4. Uniform CT
# 5. FLOWERS
# 6. BQ
# 7. RQ
# 8. SGD
aep_models = ["360 WD", "72 WD", "Average WS", "Uniform CT", "FLOWERS", "BQ", "RQ"]

def setup_wfm(deficit_model):

    wfm_base = build_wfm(site, turbines, deficit_model=deficit_model)

    wfms = []

    for aep_method in aep_models:

        if aep_method == "360 WD":
            wfm = WD_Bins(site, turbines, deficit_model, n_bins=360)
        elif aep_method == "72 WD":
            wfm = WD_Bins(site, turbines, deficit_model, n_bins=72)
        elif aep_method == "Average WS":
            wfm = average_WS(site, turbines, deficit_model)
        elif aep_method == "Uniform CT":
            wfm = uniform_CT(site, turbines, deficit_model)
        elif aep_method == "FLOWERS":
            if deficit_model == "NOJ":
                wfm = NOJ_flowers(site, turbines, n_terms=10)
            elif deficit_model == "Gaussian":
                wfm = gaussian_flowers(site, turbines, n_terms=10)
            elif deficit_model == "TurbOPark":
                wfm = TurbOPark_flowers(site, turbines, n_terms=10)
        elif aep_method == "BQ":
            wfm = setup_BQ_wfm(wfm_base, "BQ")
        elif aep_method == "RQ":
            wfm = setup_BQ_wfm(wfm_base, "RQ")
        elif aep_method == "SGD":
            wfm = SGD(site, turbines, deficit_model=deficit_model)

        wfms.append(wfm)

    return wfms


# Function to calculate and time all AEPs
def compute_aep(n_CPUs, wfms, n_repeats=5):

    times_results = []
    for wfm in wfms:
        times = []
        for _ in range(n_repeats):
            start_time = time.time()
            if wfm.name == "FLOWERS":
                AEP = wfm.aep(x, y)
            else:
                AEP = wfm.aep(x, y, n_cpu=n_CPUs)
            end_time = time.time()
            times.append(end_time - start_time)
        avg_time = np.mean(times)
        times_results.append(avg_time)

    return times_results

def compute_gradients(n_CPUs, wfms, n_repeats=5):

    times_results = []
    for wfm in wfms:
        times = []
        for _ in range(n_repeats):
            start_time = time.time()
            if wfm.name == "FLOWERS":
                AEP = wfm.aep_gradient(x, y)
            elif wfm.name in ["BQ", "RQ"]:
                wfm.setup_gradients(n_cpu=n_CPUs)
                AEP = wfm.aep_gradient(x, y)
            else:
                AEP = wfm.aep_gradient(x, y, n_cpu=n_CPUs)
            end_time = time.time()
            times.append(end_time - start_time)
        avg_time = np.mean(times)
        times_results.append(avg_time)

    return times_results

#################
# Set up all models
#################

wfm_all = {}
for deficit in ("NOJ", "Gaussian", "TurbOPark"):
    print(f"Setting up models for {deficit}...")
    wfm_all[deficit] = setup_wfm(deficit)


################# 
# Run AEP computations for different CPU counts
#################

n_CPUs_list = np.arange(1, 33)  # Test from 1 to 32 CPUs
# n_CPUs_list = [1]
results = {}
for n_CPUs in n_CPUs_list:
    results[n_CPUs] = {}
    for deficit, wfms in wfm_all.items():
        times = compute_aep(n_CPUs, wfms)
        results[n_CPUs][deficit] = dict(zip(aep_models, times))

# Save and print results in a table with all dimensions (CPU, deficit, WFM)
records = []
for n_CPUs, deficit_data in results.items():
    for deficit, wfm_data in deficit_data.items():
        for wfm_name, runtime in wfm_data.items():
            records.append({
                "n_CPUs": n_CPUs,
                "deficit": deficit,
                "wfm": wfm_name,
                "time_s": runtime,
            })

df = pd.DataFrame(records)
df.to_csv("parallelization_results_50.csv", index=False)

print(df.pivot_table(index=["deficit", "wfm"], columns="n_CPUs", values="time_s").to_string(float_format=lambda v: f"{v:.3f}"))