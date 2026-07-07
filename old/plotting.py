import math
import matplotlib.pyplot as plt
import pandas as pd

# Load the results from the parallelization experiment
df = pd.read_csv("results//Parallelization/parallelization_g_results_100.csv")

# Filter wfms (exclude 'flowers') and ensure required deficits order
wfms = [w for w in df['wfm'].unique() if w.lower() != 'flowers']
deficits = ['NOJ', 'Gaussian', 'TurbOPark']

# x values (sorted unique number of CPUs)
cpus = sorted(df['n_CPUs'].unique())

# Figure layout
n_plots = len(wfms)
ncols = 3
nrows = math.ceil(n_plots / ncols) if n_plots > 0 else 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
flat_axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

for i, w in enumerate(wfms):
    ax = flat_axes[i]
    for d in deficits:
        sel = df[(df['wfm'] == w) & (df['deficit'] == d)]
        if sel.empty:
            continue
        # compute mean time per CPU count (in case multiple runs)
        mean_times = sel.groupby('n_CPUs')['time_s'].mean().reindex(cpus)
        ax.plot(cpus, mean_times, marker='o', label=d, markersize=3)
    ax.set_title(w)
    ax.set_xlabel('Number of CPUs')
    ax.set_ylabel('Average Time (s)')
    ax.grid(True)
    ax.legend()

# Hide any unused subplots
for j in range(n_plots, len(flat_axes)):
    flat_axes[j].set_visible(False)

fig.suptitle("Parallelization results for 100 turbines", fontsize=16)
fig.tight_layout()
fig.savefig("parallelization_g_results_100.png")
plt.show()