import h5py
import numpy as np
from pathlib import Path
import pandas as pd

results_dir = Path("results/optimization")
n_starts = 50
N_BOOTSTRAP = 1000

def load_values(farm_id, aep_method):
    """Load all AEP, runtime, and n_iter values for a given farm/method."""
    aep_values, runtime_values, iter_values = [], [], []
    filepath = results_dir / f"windfarm_{farm_id}.h5"

    with h5py.File(filepath, "r") as f:
        for start_idx in range(n_starts):
            group_key = f"{aep_method}/Gaussian/start_{start_idx}"
            if group_key not in f:
                continue
            grp = f[group_key]
            aep_values.append(grp.attrs.get("aep_final", np.nan))
            runtime_values.append(grp.attrs.get("time", np.nan))
            iter_values.append(grp.attrs.get("n_iter", np.nan))

    return np.array(aep_values), np.array(runtime_values), np.array(iter_values)


def bootstrap_median_errors(values, sample_sizes, n_bootstrap=N_BOOTSTRAP):
    """
    For each sample size, draw n_bootstrap random subsets (without replacement)
    and compute the median of each. Returns mean abs % error and std of % error
    relative to the full-sample median (baseline).
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return {n: {"mean_abs_pct_err": np.nan, "std_pct_err": np.nan,
                    "median_of_medians": np.nan} for n in sample_sizes}

    baseline_median = np.nanmedian(valid)
    results = {}

    for n in sample_sizes:
        if n >= len(valid):
            results[n] = {"mean_abs_pct_err": 0.0, "std_pct_err": 0.0,
                          "median_of_medians": baseline_median}
            continue

        subsample_medians = np.array([
            np.median(valid[np.random.choice(len(valid), size=n, replace=False)])
            for _ in range(n_bootstrap)
        ])

        pct_errors = (subsample_medians - baseline_median) / baseline_median * 100
        results[n] = {
            "mean_abs_pct_err": np.mean(np.abs(pct_errors)),
            "std_pct_err": np.std(pct_errors),
            "median_of_medians": np.median(subsample_medians),
        }

    return results


def iter_summary(iter_values):
    """Summarize n_iter distribution: median, min, max, and std."""
    valid = iter_values[~np.isnan(iter_values)]
    if len(valid) == 0:
        return {"median": np.nan, "min": np.nan, "max": np.nan, "std": np.nan}
    return {
        "median": int(np.median(valid)),
        "min":    int(np.min(valid)),
        "max":    int(np.max(valid)),
        "std":    np.std(valid),
    }


farms = ["East_Anglia_TWO", "Thor", "Sofia"]
methods = ["FLOWERS", "72_WD", "Average_WS", "Uniform_CT", "BQ", "360_WD", "RQ"]
sample_sizes = [5, 10, 20, 30, 40, 50]

for farm in farms:
    print(f"\n{'='*80}")
    print(f"Farm: {farm}")
    print('='*80)

    aep_rows, runtime_rows, iter_rows = [], [], []

    for method in methods:
        aep_vals, runtime_vals, iter_vals = load_values(farm, method)

        aep_stats     = bootstrap_median_errors(aep_vals,     sample_sizes)
        runtime_stats = bootstrap_median_errors(runtime_vals, sample_sizes)
        its           = iter_summary(iter_vals)

        aep_row     = {"method": method}
        runtime_row = {"method": method}
        iter_row    = {
            "method":        method,
            "median_iter":   its["median"],
            "min_iter":      its["min"],
            "max_iter":      its["max"],
            "std_iter":      f"{its['std']:.1f}",
            "range":         f"{its['min']}–{its['max']}",
        }

        for n in sample_sizes:
            a = aep_stats[n]
            r = runtime_stats[n]
            aep_row[f"n={n}"]     = f"{a['mean_abs_pct_err']:.3f}% ± {a['std_pct_err']:.3f}%"
            runtime_row[f"n={n}"] = f"{r['mean_abs_pct_err']:.3f}% ± {r['std_pct_err']:.3f}%"

        aep_rows.append(aep_row)
        runtime_rows.append(runtime_row)
        iter_rows.append(iter_row)

    print("\n--- AEP: Mean |% error| ± std of % error (vs. 50-sample median baseline) ---")
    print(pd.DataFrame(aep_rows).set_index("method").to_string())

    print("\n--- Runtime: Mean |% error| ± std of % error (vs. 50-sample median baseline) ---")
    print(pd.DataFrame(runtime_rows).set_index("method").to_string())

    print("\n--- Iteration count distribution across 50 starts ---")
    iter_df = pd.DataFrame(iter_rows).set_index("method")[
        ["median_iter", "min_iter", "max_iter", "std_iter", "range"]
    ]
    print(iter_df.to_string())


# Plot the median AEP and runtime errors for each method across sample sizes for all wind farms
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
for i, farm in enumerate(farms):
    ax_aep = axes[0, i]
    ax_runtime = axes[1, i]

    for method in methods:
        aep_vals, runtime_vals, _ = load_values(farm, method)
        aep_stats = bootstrap_median_errors(aep_vals, sample_sizes)
        runtime_stats = bootstrap_median_errors(runtime_vals, sample_sizes)

        aep_series = [aep_stats[n]['mean_abs_pct_err'] for n in sample_sizes]
        runtime_series = [runtime_stats[n]['mean_abs_pct_err'] for n in sample_sizes]

        ax_aep.plot(sample_sizes, aep_series, marker='o', label=method)
        ax_runtime.plot(sample_sizes, runtime_series, marker='x', linestyle='--', label=method)

    ax_aep.set_title(f"{farm} - AEP")
    ax_runtime.set_title(f"{farm} - Runtime")
    ax_runtime.set_xlabel("Sample Size")
    ax_aep.set_ylabel("Mean |% Error|")
    ax_runtime.set_ylabel("Mean |% Error|")
    ax_aep.legend()
    ax_runtime.legend()
    ax_aep.grid()
    ax_runtime.grid()
plt.tight_layout()
plt.savefig("median_errors_across_farms_30.png")