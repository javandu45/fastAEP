from FAST_AEP.BQ import bayesian_quadrature
from FAST_AEP.utils import generic_site, generate_random_array

from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.deficit_models import TurboNOJDeficit
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


SITE = generic_site("Dogger_Bank_C")
TURBINES = V80()
X0, Y0 = Hornsrev1Site().initial_position.T  # reference layout for GP training

def _random_square_layout(n_tur: int, spacing: int):
    side = int(np.sqrt(n_tur)) * spacing * TURBINES.diameter() * 1.5  # add some extra space to avoid edge effects
    limits = np.array([(0, 0), (side, 0), (side, side), (0, side)])
    return generate_random_array(n_tur=n_tur, turbine=TURBINES, spacing=spacing, limits=limits)

X0, Y0 = _random_square_layout(200, spacing=4)  # 100 turbines with 4 rotor diameters spacing

def _build_flow_model(deficit_model: str):
    if deficit_model == "NOJ":
        return Jensen_1983(SITE, TURBINES)
    if deficit_model == "Gaussian":
        return Bastankhah_PorteAgel_2014(SITE, TURBINES, k=0.05)
    if deficit_model == "TurbOPark":
        return PropagateDownwind(
            SITE, TURBINES,
            wake_deficitModel=TurboNOJDeficit(),
            superpositionModel=SquaredSum(),
            rotorAvgModel=AreaOverlapAvgModel(),
        )
    raise ValueError(f"Unknown deficit model: {deficit_model}")


def setup_model(deficit_model: str, aep_method: str, n_train: int = 100) -> bayesian_quadrature:
    wfm = _build_flow_model(deficit_model)
    model = bayesian_quadrature(
        site=SITE,
        windTurbines=TURBINES,
        flow_model=wfm,
        x0=X0,
        y0=Y0,
        N_train=n_train,
        N_MC=4000,
        aep_method=aep_method,
    )
    model.train_and_get_kernel()
    model.optimize_BQ_points(N_points=360, N_attempts=1)
    return model


def setup_all_models(n_train: int = 100) -> dict:
    models = {}
    for deficit in ("NOJ", "Gaussian", "TurbOPark"):
        for method in ("BQ", "RQ"):
            key = f"{method}_{deficit}"
            print(f"Setting up {key}...")
            models[key] = setup_model(deficit, method, n_train)
    return models


# ── Experiment 1: time vs. number of CPUs ────────────────────────────────────

def benchmark_cpu(models: dict, x, y, n_repeats: int = 10, cpu_counts=(1, 4, 8, 16)):
    results = {}
    for n_cpu in cpu_counts:
        row = {}
        for name, model in models.items():
            times = [
                _time_aep(model, x, y, n_cpu)
                for _ in range(n_repeats)
            ]
            row[name] = np.mean(times)
        results[f"{n_cpu} CPU{'s' if n_cpu > 1 else ''}"] = row

    df = pd.DataFrame(results).T
    df.index.name = "CPUs"
    print("\nAverage AEP computation time (seconds)\n")
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))


# ── Experiment 2: time vs. number of turbines for different CPU counts ────────

def benchmark_n_turbines(
    models: dict,
    turbine_counts=(50, 100, 150, 200, 250, 300),
    cpu_counts=(1, 4, 8, 16),
    n_repeats: int = 5,
    spacing: int = 4,
):
    # results[(model_name, n_cpu)][n_turbines] = avg_time
    results = {(name, n_cpu): {} for name in models for n_cpu in cpu_counts}

    for n_tur in turbine_counts:
        x, y = _random_square_layout(n_tur, spacing)
        print(f"Timing {n_tur} turbines...")
        for name, model in models.items():
            for n_cpu in cpu_counts:
                times = [_time_aep(model, x, y, n_cpu) for _ in range(n_repeats)]
                results[(name, n_cpu)][n_tur] = np.mean(times)

    _plot_turbine_benchmark(results, turbine_counts, cpu_counts)


def _plot_turbine_benchmark(results: dict, turbine_counts, cpu_counts):
    deficit_models = ("NOJ", "Gaussian", "TurbOPark")
    methods = ("BQ", "RQ")
    cpu_colors = plt.cm.viridis(np.linspace(0, 0.85, len(cpu_counts)))

    fig, axes = plt.subplots(
        len(deficit_models), len(methods),
        figsize=(12, 4 * len(deficit_models)),
        sharey="row", sharex=True,
    )

    for row, deficit in enumerate(deficit_models):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            key_base = f"{method}_{deficit}"
            for n_cpu, color in zip(cpu_counts, cpu_colors):
                times = [results[(key_base, n_cpu)][n] for n in turbine_counts]
                ax.plot(turbine_counts, times, marker="o", color=color,
                        label=f"{n_cpu} CPU{'s' if n_cpu > 1 else ''}")
            ax.set_title(f"{deficit} — {method}")
            ax.grid(True, linestyle="--", alpha=0.5)
            if col == 0:
                ax.set_ylabel("Avg computation time (s)")
            if row == len(deficit_models) - 1:
                ax.set_xlabel("Number of turbines")

    # Single shared legend on the last axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="CPU count", loc="lower center",
               ncol=len(cpu_counts), bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("AEP computation time vs. number of turbines", y=1.01)
    plt.tight_layout()
    plt.savefig("aep_benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()


# ── Shared helper ─────────────────────────────────────────────────────────────

def _time_aep(model, x, y, n_cpu: int = 1) -> float:
    t0 = time.perf_counter()
    model.aep(x, y, n_cpu=n_cpu)
    return time.perf_counter() - t0


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    models = setup_all_models(n_train=100)

    print("\n=== Experiment 1: time vs. CPUs ===")
    benchmark_cpu(models, X0, Y0)

    # print("\n=== Experiment 2: time vs. number of turbines ===")
    # benchmark_n_turbines(models)