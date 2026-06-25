import h5py
from pathlib import Path
import pandas as pd
import numpy as np

farm_id = "Hornsea_Project_3_HOW03"
aep_method = "FLOWERS"
wake_model = "Gaussian"
median = False
n_starts = 5

results_dir = Path("results/optimization")
h5_path = results_dir / f"windfarm_{farm_id}.h5"

records = []
median_records = []

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

# Print table with results
results_table = pd.DataFrame(records)
print(f"Results for {farm_id} with {aep_method} and {wake_model}:")
print(results_table.round(2).to_string(index=False))