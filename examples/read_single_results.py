

# A code to read and summarize results from AEP calculations with different methods,
# wind farms, and wake models.

import h5py
from pathlib import Path
import pandas as pd
import numpy as np

farm_id = "Hornsea_Project_2_-_Phase_1_Breesea"
farm_id = "Hornsea_Project_3_HOW03"
aep_method = "SGD"
wake_model = "NOJ"
median = True
n_starts = 30

results_dir = Path("results/optimization/Iteration_3_4WF_30starts")
# results_dir = Path("results/optimization")
h5_path = results_dir / f"windfarm_{farm_id}.h5"
# h5_path = "windfarm_Hornsea_Project_3_HOW03.h5"

records = []
median_records = []

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

with h5py.File(h5_path, "r") as f:

    for start_idx in range(n_starts):
        group_key = f"{aep_method}/{wake_model}/start_{start_idx}"
        if group_key not in f:
            continue

        grp = f[group_key]
        records.append({
            "start_idx": start_idx,
            "aep": grp.attrs.get("aep_final", np.nan),
            "time": grp.attrs.get("time", np.nan),
            "iteration": grp.attrs.get("n_iter", np.nan),
        })

        cost = grp["convergence"][:]

        plt.plot(-cost, label=f"Start {start_idx}", alpha=0.5)

plt.title(f"Convergence for {farm_id} with {aep_method} and {wake_model}")
plt.xlabel("Iteration")
plt.ylabel("Convergence")
# plt.savefig(f"convergence_{farm_id}_{aep_method}_{wake_model}.png", dpi=300)

# Print medians
if median:
    records_df = pd.DataFrame(records)
    median_aep = records_df["aep"].median()
    median_time = records_df["time"].median()
    median_iteration = records_df["iteration"].median()

    median_records.append({
        "start_idx": "Median",
        "aep": median_aep,
        "time": median_time,
        "iteration": median_iteration,
    })

    records = pd.DataFrame(median_records)


# Print table with results
results_table = pd.DataFrame(records)
print(f"Results for {farm_id} with {aep_method} and {wake_model}:")
print(results_table.round(2).to_string(index=False))

