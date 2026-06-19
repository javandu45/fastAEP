import h5py
import pandas as pd
from pathlib import Path

wake_models = ["NOJ"]
wf_data = pd.read_csv("data/top_20_windfarms.csv", index_col=0)
wind_farms = wf_data['name'].tolist()

results_dir = Path("results/optimization")

for wake_model in wake_models:

    # Collect rows as (farm_id, aep_method) -> value, then pivot
    records = []

    for farm_id in wind_farms:
        if farm_id in ("Kriegers_Flak_K2-K3", "Gwynt_y_Mor"):
            continue

        h5_path = results_dir / f"windfarm_{farm_id}.h5"
        if not h5_path.exists():
            print(f"Missing file, skipping: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as f:
            for aep_method in f.keys():
                group_key = f"{aep_method}/{wake_model}/start_0"
                if group_key not in f:
                    continue  # this method/wake combo hasn't been run for this farm yet

                grp = f[group_key]
                records.append({
                    "farm_id":    farm_id,
                    "aep_method": aep_method,
                    "aep":        grp.attrs["aep_final"],
                    "runtime":    grp.attrs["time"],
                    "success":    bool(grp.attrs["success"]),
                })

    if not records:
        print(f"No results found for wake model {wake_model}")
        continue

    df = pd.DataFrame(records)

    # Pivot: rows = farm_id, columns = aep_method
    aep_table     = df.pivot(index="farm_id", columns="aep_method", values="aep")
    runtime_table = df.pivot(index="farm_id", columns="aep_method", values="runtime")
    success_table = df.pivot(index="farm_id", columns="aep_method", values="success")

    # Keep wind farm row order consistent with the CSV
    farm_order = [f for f in wind_farms if f in aep_table.index]
    aep_table     = aep_table.loc[farm_order]
    runtime_table = runtime_table.loc[farm_order]
    success_table = success_table.loc[farm_order]

    print(f"\n{'='*70}")
    print(f"Wake model: {wake_model}")
    print(f"{'='*70}")

    print("\n--- AEP [GWh] ---")
    print(aep_table.round(3).to_string())

    print("\n--- Runtime [s] ---")
    print(runtime_table.round(1).to_string())

    print("\n--- Success ---")
    print(success_table.to_string())

    # Optional: save to disk for later use / Excel inspection
    aep_table.to_csv(results_dir / f"summary_aep_{wake_model}.csv")
    runtime_table.to_csv(results_dir / f"summary_runtime_{wake_model}.csv")
    success_table.to_csv(results_dir / f"summary_success_{wake_model}.csv")

