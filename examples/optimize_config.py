import argparse
import json
import time
from FAST_AEP.utils import build_aep_model, save_results_in_H5
from FAST_AEP.optimization import optifast

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--task-file", required=True)
parser.add_argument("--task-id",   required=True, type=int)
args = parser.parse_args()

########################################################################################
# Extract configuration for the current job
with open(args.task_file) as f:
    tasks = json.load(f)
task = tasks[args.task_id]

farm_id = task["farm_id"]
aep_method = task["aep_method"]
wake_model = task["wake_model"]
n_cpu = task["n_cpu"]
start_id = task["start_id"]
tol = task["tol"]

print("Optimizing configuration:")
print(f"Farm: {farm_id}")
print(f"AEP method: {aep_method}")
print(f"Wake model: {wake_model}")

# Set up wind farm model
wfm = build_aep_model(aep_method=aep_method, wind_farm=farm_id, deficit_model=wake_model, n_cpu=n_cpu)

# Set up evaluating model
evaluating_wfm = build_aep_model(aep_method="360_WD", wind_farm=farm_id, deficit_model=wake_model, n_cpu=n_cpu)

########################################################################################
# Set up optimization
# Excpected costs
if wake_model == "NOJ":
    expected_cost = 10
elif wake_model == "Gaussian":
    expected_cost = 1
    if aep_method == "FLOWERS":
        expected_cost = 1e-3
elif wake_model == "TurbOPark":
    expected_cost = 10

# Maximum number of iterations
max_iter = 150
if (aep_method == "RQ" or aep_method == "BQ") and wake_model == "Gaussian":
    max_iter = 100

# Set up optimization problem
optimization_problem = optifast(wind_farm=farm_id, wind_farm_model=wfm, min_spacing=None, n_cpu=n_cpu, seed=start_id)

tf_problem = optimization_problem.setup_problem(tolerance=tol, expected_cost=expected_cost, max_iter=max_iter)

print("Running optimization...")
time_start = time.time()
_, state, recorder = tf_problem.optimize()
time_end = time.time()
time_1 = time_end - time_start

########################################################################################
# Convert from normalized to actual coordinates if needed
if optimization_problem.normalization:
    x_opt = state["x_norm"] * optimization_problem.max_x
    y_opt = state["y_norm"] * optimization_problem.max_y

else:
    x_opt = state["x"]
    y_opt = state["y"]

final_aep = evaluating_wfm.aep(x_opt, y_opt)

########################################################################################
# Save results in HDF5 file
res = {
    "x_opt": x_opt,
    "y_opt": y_opt,
    "convergence": recorder.get("cost"),
    "aep_final": final_aep,
    "time": time_1,
    "n_iter": len(recorder.get("cost")),
    "success": tf_problem.driver.result.success
}

print(f"Optimization completed in {time_1:.2f} seconds")
print(f"Optimized AEP: {final_aep:.2f} GWh")

save_results_in_H5(farm_id=farm_id,
                   aep_method=aep_method,
                   wake_model=wake_model,
                   start_id=start_id,
                   res=res)

