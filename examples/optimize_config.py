import argparse
import json
import time
from FAST_AEP.utils import build_aep_model
from FAST_AEP.optimization import optifast

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

print("Optimizing configuration:")
print(f"Farm: {farm_id}")
print(f"AEP method: {aep_method}")
print(f"Wake model: {wake_model}")

# Set up wind farm model
wfm = build_aep_model(aep_method=aep_method, wind_farm=farm_id, deficit_model=wake_model)

# Set up evaluating model
evaluating_wfm = build_aep_model(aep_method="360_WD", wind_farm=farm_id, deficit_model=wake_model)

########################################################################################
# Set up optimization
optimization_problem = optifast(wind_farm=farm_id, wind_farm_model=wfm, min_spacing=None, n_cpu=n_cpu)

tf_problem = optimization_problem.setup_problem(tolerance=1e-6, expected_cost=10, max_iter=100)

time_start = time.time()
_, state, recorder = tf_problem.optimize()
time_end = time.time()
time_1 = time_end - time_start

if optimization_problem.normalization:
    x_opt = state["x_norm"] * optimization_problem.max_x
    y_opt = state["y_norm"] * optimization_problem.max_y

else:
    x_opt = state["x"]
    y_opt = state["y"]

final_aep = evaluating_wfm.aep(x_opt, y_opt)

print(f"Optimization completed in {time_1:.2f} seconds")
print(f"Optimized AEP: {final_aep:.2f} GWh")

