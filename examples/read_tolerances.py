import h5py
import pandas as pd
from pathlib import Path
import numpy as np

wake_models = ["Gaussian"]
wf_data = pd.read_csv("data/top_20_windfarms.csv", index_col=0)
wind_farms = wf_data['name'].tolist()
wind_farms = ["Dogger_Bank_B"]

results_dir = Path("results/optimization")

import matplotlib.pyplot as plt

for wake_model in wake_models:

    # Collect rows as (farm_id, aep_method) -> value, then pivot
    records = []
    # convergence data: keyed by (farm_id, aep_method) -> list of 3 convergence arrays
    conv_data = {}
    # final AEP values keyed by (farm_id, aep_method)
    aep_data = {}

    for farm_id in wind_farms:
        if farm_id in ("Kriegers_Flak_K2-K3", "Gwynt_y_Mor"):
            continue

        h5_path = results_dir / f"windfarm_{farm_id}.h5"
        if not h5_path.exists():
            print(f"Missing file, skipping: {h5_path}")
            continue

        

        def _print_h5_structure(hf):
            def _recurse(name, obj):
                indent = '  ' * (name.count('/') )
                if isinstance(obj, h5py.Group):
                    print(f"{indent}{name.split('/')[-1] or '/'} (group)")
                else:
                    print(f"{indent}{name.split('/')[-1]} (dataset)")

            print(f"HDF5 structure for {h5_path}:")
            hf.visititems(_recurse)

        with h5py.File(h5_path, "r") as f:
            # print the file/group/dataset structure to inspect contents
            # _print_h5_structure(f)
            for aep_method in f.keys():
                # gather three starts
                convs = []
                any_start = False
                for s in [5, 6, 7]:
                    group_key = f"{aep_method}/{wake_model}/start_{s}"
                    if group_key not in f:
                        convs.append(None)
                        # keep alignment with convs: aep for this start is missing
                        # will be stored as None below
                        if 'aep_list' in locals():
                            aep_list.append(None)
                        else:
                            # ensure aep_list exists before first append
                            aep_list = [None] * (len(convs)-1) + [None]
                        continue

                    any_start = True
                    # ensure aep_list exists
                    if 'aep_list' not in locals():
                        aep_list = []
                    grp = f[group_key]
                    aep_val = grp.attrs.get("aep_final", float('nan'))
                    records.append({
                        "farm_id":    farm_id,
                        "aep_method": aep_method,
                        "aep":        aep_val,
                        "runtime":    grp.attrs.get("time", float('nan')),
                        "success":    bool(grp.attrs.get("success", False)),
                    })

                    aep_list.append(aep_val)

                    # try to extract convergence series from dataset or attr
                    conv = None
                    if "convergence" in grp:
                        try:
                            conv = grp["convergence"][:]
                        except Exception:
                            conv = None
                    else:
                        # maybe stored as attr
                        conv = grp.attrs.get("convergence", None)

                    # ensure numpy array or list
                    if conv is not None:
                        try:
                            import numpy as _np
                            conv = _np.asarray(conv).tolist()
                        except Exception:
                            pass

                    convs.append(conv)

                if not any_start:
                    continue

                # ensure aep_list has length 3
                if 'aep_list' not in locals():
                    aep_list = [None, None, None]
                else:
                    # pad or trim to length 3
                    aep_list = (aep_list + [None, None, None])[:3]

                conv_data[(farm_id, aep_method)] = convs
                aep_data[(farm_id, aep_method)] = aep_list
                # cleanup local aep_list for next aep_method
                del aep_list

    if not records:
        print(f"No results found for wake model {wake_model}")
        continue

    # df = pd.DataFrame(records)

    # # Pivot: rows = farm_id, columns = aep_method
    # aep_table     = df.pivot(index="farm_id", columns="aep_method", values="aep")
    # runtime_table = df.pivot(index="farm_id", columns="aep_method", values="runtime")
    # success_table = df.pivot(index="farm_id", columns="aep_method", values="success")

    # # Keep wind farm row order consistent with the CSV
    # farm_order = [f for f in wind_farms if f in aep_table.index]
    # aep_table     = aep_table.loc[farm_order]
    # runtime_table = runtime_table.loc[farm_order]
    # success_table = success_table.loc[farm_order]

    # print(f"\n{'='*70}")
    # print(f"Wake model: {wake_model}")
    # print(f"{'='*70}")

    # print("\n--- AEP [GWh] ---")
    # print(aep_table.round(3).to_string())

    # print("\n--- Runtime [s] ---")
    # print(runtime_table.round(1).to_string())

    # print("\n--- Success ---")
    # print(success_table.to_string())

    # # Optional: save to disk for later use / Excel inspection
    # aep_table.to_csv(results_dir / f"summary_aep_{wake_model}.csv")
    # runtime_table.to_csv(results_dir / f"summary_runtime_{wake_model}.csv")
    # success_table.to_csv(results_dir / f"summary_success_{wake_model}.csv")

    # Make convergence figures for each farm / aep_method
    results_dir.mkdir(parents=True, exist_ok=True)
    for (farm_id, aep_method), convs in conv_data.items():
        # only plot if at least one convergence curve exists
        if not any(c is not None for c in convs):
            continue

        aep_list = aep_data.get((farm_id, aep_method), [float('nan')] * 3)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
        for i in range(3):
            ax = axes[i]
            conv = convs[i]
            if conv is None:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center')
                ax.set_title(f'start_{i+5}')
                ax.set_xlim(0, 1)
                continue
            ax.plot(-np.array(conv), marker='o', linewidth=1.5)
            aep_final = aep_list[i]
            aep_str = f"{aep_final:.3f}" if aep_final is not None and not (isinstance(aep_final, float) and np.isnan(aep_final)) else 'nan'
            ax.set_title(f'Tolerance=1e-{5+i}, AEP={aep_str}')
            ax.set_xlabel('iter')
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel('convergence')
        fig.suptitle(f"{farm_id} | {aep_method} | {wake_model}")
        out_path = results_dir / f"convergence_{farm_id}_{aep_method}_{wake_model}.png"
        fig.savefig(out_path)
        plt.close(fig)

