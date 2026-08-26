#%%
from FAST_AEP.BQ import bayesian_quadrature
from FAST_AEP.utils import generic_site
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.noj import Jensen_1983

site = generic_site("Hornsea_Project_3_HOW03")
x, y = Hornsrev1Site().initial_position.T

wfm_ref = Jensen_1983(site=site, windTurbines=V80(), k=0.05)
aep_ref = wfm_ref.aep(x, y)
print(f"Reference AEP: {aep_ref:.2f} GWh")

wfm = bayesian_quadrature(site=site,
                          windTurbines=V80(),
                          flow_model=wfm_ref,
                          N_train=3000,
                          N_MC=4000,
                          x0=x, y0=y)

wfm.train_and_get_kernel()

# %%
N_runs = 20
N_attempts = np.arange(1, 51, 2)
AEPs = np.zeros((N_runs, len(N_attempts)))

for run in range(N_runs):
    print(f"\n=== Run {run + 1}/{N_runs} ===")
    for i, N in enumerate(N_attempts):
        print(f"  Multistarts: {N}...")
        wfm.optimize_BQ_points(N_points=360, N_attempts=N, tol=1e-8, jitter=1e-2)
        AEPs[run, i] = wfm.aep(x, y)
        print(f"  AEP: {AEPs[run, i]:.4f} GWh")

# Save raw results
results_df = pd.DataFrame(AEPs, columns=[f'N={n}' for n in N_attempts])
results_df.to_csv('BQ_points_results.csv', index=False)
print("\nResults saved to BQ_points_results.csv")

# %%
mean_aep = AEPs.mean(axis=0)
std_aep = AEPs.std(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax = axes[0]
ax.plot(N_attempts, mean_aep, marker='o', label='Mean AEP')
ax.fill_between(N_attempts,
                mean_aep - std_aep,
                mean_aep + std_aep,
                alpha=0.3, label='±1 std')
ax.axhline(aep_ref, color='r', linestyle='--', label='Reference AEP')
ax.set_ylabel('AEP (GWh)')
ax.set_title(f'AEP vs Number of Multistarts ({N_runs} runs)')
ax.legend()

ax = axes[1]
ax.plot(N_attempts, std_aep, marker='o', color='orange', label='Std')
# Exponential decay to a floor: std ~ a * exp(-b * N) + c
from scipy.optimize import curve_fit
def exp_floor(N, a, b, c):
    return a * np.exp(-b * N) + c
try:
    p0 = [std_aep[0] - std_aep[-1], 0.1, std_aep[-1]]
    popt, _ = curve_fit(exp_floor, N_attempts, std_aep, p0=p0, maxfev=5000)
    trend = exp_floor(N_attempts, *popt)
    ax.plot(N_attempts, trend, '--', color='gray',
            label=f'Fit: $a e^{{-bN}} + c$,  $c={popt[2]:.2f}$ GWh')
except RuntimeError:
    pass  # fit failed, skip trend line
ax.set_xlabel('Number of Multistarts')
ax.set_ylabel('Std of AEP (GWh)')
ax.set_title('Variability across runs')
ax.legend()

plt.tight_layout()
plt.savefig('BQ_points_convergence.png', dpi=150)
plt.show()
