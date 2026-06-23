import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

wake_models = ["Gaussian"]
wf_data = pd.read_csv("data/top_20_windfarms.csv", index_col=0)
wind_farms = wf_data['name'].tolist()
wind_farms.remove("Kriegers_Flak_K2-K3")  # Remove for now, since it is two different boundaries
wind_farms.remove("Gwynt_y_Mor")

# For now, only 10 wind farms
wind_farms = wind_farms[:10]

results_dir = Path("results/optimization")

# Normalize with respect to the highest AEP achieved for each farm/wake combo, and the lowest runtime (for better readability in tables)
normalize_results = False

# Print table with results
print_table = False

# Print median
median = False

n_starts = 5

# #################################################################################

for wake_model in wake_models:

    # Collect rows as (farm_id, aep_method) -> value, then pivot
    records = []
    median_records = []

    for farm_id in wind_farms:
        if farm_id in ("Kriegers_Flak_K2-K3", "Gwynt_y_Mor"):
            continue

        h5_path = results_dir / f"windfarm_{farm_id}.h5"
        if not h5_path.exists():
            print(f"Missing file, skipping: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as f:
            for aep_method in f.keys():
                # Collect all multi-start runs for this farm/method/wake combo
                aep_values = []
                runtime_values = []
                iteration_values = []
                start_idx = 0
                
                for start_idx in range(n_starts):
                    group_key = f"{aep_method}/{wake_model}/start_{start_idx}"
                    if group_key not in f:
                        continue  # No more starts
                    
                    grp = f[group_key]
                    # Some starts may lack attributes; use NaN for missing values
                    aep_values.append(grp.attrs.get("aep_final", np.nan))
                    runtime_values.append(grp.attrs.get("time", np.nan))
                    iteration_values.append(grp.attrs.get("n_iter", np.nan))
                    start_idx += 1
                
                # Process first start for original table
                if aep_values:
                    records.append({
                        "farm_id":    farm_id,
                        "aep_method": aep_method,
                        "aep":        aep_values[0],
                        "runtime":    runtime_values[0],
                        "iteration":  iteration_values[0],
                    })
                    
                    if median:

                        # Compute medians for median table
                        median_records.append({
                            "farm_id":    farm_id,
                            "aep_method": aep_method,
                            "aep":        pd.Series(aep_values).median(skipna=True),
                            "runtime":    pd.Series(runtime_values).median(skipna=True),
                            "iteration":  pd.Series(iteration_values).median(skipna=True),
                        })

    if not records:
        print(f"No results found for wake model {wake_model}")
        continue

# ##########################################################################################

    df = pd.DataFrame(records)

    # Pivot: rows = farm_id, columns = aep_method
    aep_table     = df.pivot(index="farm_id", columns="aep_method", values="aep")
    runtime_table = df.pivot(index="farm_id", columns="aep_method", values="runtime")
    iteration_table = df.pivot(index="farm_id", columns="aep_method", values="iteration")

    # Keep wind farm row order consistent with the CSV
    farm_order = [f for f in wind_farms if f in aep_table.index]
    aep_table     = aep_table.loc[farm_order]
    runtime_table = runtime_table.loc[farm_order]
    iteration_table = iteration_table.loc[farm_order]



    if print_table:

        if normalize_results:
            # Normalize AEP with respect to the AEP_max - AEP_min achieved for each farm/wake combo
            aep_table = aep_table.sub(aep_table.min(axis=1), axis=0).div(aep_table.max(axis=1).sub(aep_table.min(axis=1)), axis=0)

            # Normalize runtime with respect to the lowest runtime (for better readability in tables)
            runtime_table = runtime_table.div(runtime_table.min(axis=1), axis=0)

        print(f"\n{'='*70}")
        print(f"Wake model: {wake_model}")
        print(f"{'='*70}")

        print("\n--- AEP [GWh] ---")
        print(aep_table.round(3).to_string())

        print("\n--- Runtime [s] ---")
        print(runtime_table.round(1).to_string())

        print("\n--- Iteration ---")
        print(iteration_table.to_string())

        # Optional: save to disk for later use / Excel inspection
        aep_table.to_csv(results_dir / f"summary_aep_{wake_model}.csv")
        runtime_table.to_csv(results_dir / f"summary_runtime_{wake_model}.csv")
        iteration_table.to_csv(results_dir / f"summary_iteration_{wake_model}.csv")

    if median:
        median_df = pd.DataFrame(median_records)

        median_aep_table     = median_df.pivot(index="farm_id", columns="aep_method", values="aep")
        median_runtime_table = median_df.pivot(index="farm_id", columns="aep_method", values="runtime")
        median_iteration_table = median_df.pivot(index="farm_id", columns="aep_method", values="iteration")

        # Keep wind farm row order consistent with the CSV
        median_aep_table     = median_aep_table.loc[farm_order]
        median_runtime_table = median_runtime_table.loc[farm_order]
        median_iteration_table = median_iteration_table.loc[farm_order]

        if normalize_results:
            # Normalize AEP with respect to the AEP_max - AEP_min achieved for each farm/wake combo
            median_aep_table = median_aep_table.sub(median_aep_table.min(axis=1), axis=0).div(median_aep_table.max(axis=1).sub(median_aep_table.min(axis=1)), axis=0)

            # Normalize runtime with respect to the lowest runtime (for better readability in tables)
            median_runtime_table = median_runtime_table.div(median_runtime_table.min(axis=1), axis=0)

        print(f"\n{'='*70}")
        print(f"Median results - Wake model: {wake_model}")
        print(f"{'='*70}")

        print("\n--- Median AEP [GWh] ---")
        print(median_aep_table.round(3).to_string())

        print("\n--- Median Runtime [s] ---")
        print(median_runtime_table.round(1).to_string())

        print("\n--- Median Iteration ---")
        print(median_iteration_table.to_string())

# ########################################################################################
# plot convergence plot for 360_WD for Hornsea_Project_3_HOW03
plt.figure(figsize=(8, 5))
with h5py.File(results_dir / "windfarm_Hornsea_Project_3_HOW03.h5", "r") as f:
    base_key = "FLOWERS/Gaussian"
    # Loop through all starts and plot each convergence curve
    start_idx = 0
    found = False
    while True:
        group_key = f"{base_key}/start_{start_idx}"
        if group_key not in f:
            break
        grp = f[group_key]
        if "convergence" in grp:
            convergence = grp["convergence"][:]
            plt.plot(-convergence, label=f"start_{start_idx}")
            found = True
        start_idx += 1
    if not found:
        print(f"No convergence data found under {base_key} in file, cannot plot convergence")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Convergence plot for Hornsea_Project_3_HOW03 (360_WD, Gaussian)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("convergence_Hornsea_Project_3_HOW03_360_WD_NOJ.png")
plt.show()

