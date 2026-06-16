
# This file aims to compare the performance of PyWake against FLOWERS and BQ to determine
# their suitability for fast optimization.

# The test will be ran for a 9 turbine case, with 10 random starts, and a maximum of 100 iterations.
# The chosen flow model is NO Jensen, and all the final layouts evaluated with PyWake.

# Import fast AEP
from FAST_AEP.BQ import bayesian_quadrature
from FAST_AEP.FLOWERS import NOJ_flowers

# Ignore numerical issues from FLOWERS
import warnings
warnings.filterwarnings("ignore")

# PyWake imports
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.noj import Jensen_1983
from py_wake.utils.gradients import autograd

# TopFarm imports
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver

# Other imports
from utils import generate_random_array, generic_site
from matplotlib import pyplot as plt
import time
import pandas as pd
import numpy as np

# Set up problem conditions
site = generic_site(ws=10)
turbine = V80()
N_turbines = 20

# Create squared limits based on number of turbines and spacing
turbines_per_row = int(np.sqrt(N_turbines))
sep = turbines_per_row*6*turbine.diameter()
limits = np.array([(0, 0), (sep, 0), (sep,sep), (0, sep)])
min_dist = 5*turbine.diameter()

# Generate intial random layout based on the limits, number of turbines and spacing - Needed to train BQ model
x_init, y_init = generate_random_array(n_tur=N_turbines, turbine=turbine, spacing=4, limits=limits)

####################
# Set up AEP models
####################

# PyWake
pw_wfm = Jensen_1983(site=site, windTurbines=turbine, k=0.04)

# Bayesian Quadrature
BQ_wfm = bayesian_quadrature(site=site,
                            windTurbines=turbine,
                            flow_model=pw_wfm,
                            x0=x_init,
                            y0=y_init,
                            N_train=3000,
                            N_MC=4000,
                            aep_method="BQ",
                            kernel_type="linear")

BQ_wfm.train_and_get_kernel()

# FLOWERS
flowers_wfm = NOJ_flowers(site=site, windTurbines=turbine, k=0.04, n_terms=10)

# Pseudo-BQ 
pseudo_BQ_wfm = bayesian_quadrature(site=site,
                            windTurbines=turbine,
                            flow_model=pw_wfm,
                            x0=x_init,
                            y0=y_init,
                            N_train=3000,
                            N_MC=4000,
                            aep_method="MC",
                            kernel_type="linear")

pseudo_BQ_wfm.train_and_get_kernel()

####################
# Function to evaluate final AEP

def final_aep_function(x, y):
    sim_res = pw_wfm(x, y,
            wd=site.default_wd,
            ws=site.default_ws)
    
    return sim_res.aep().sum().values

####################
# Set up optimization problem
####################

# Set up constraints
turb_sep_constraint = SpacingConstraint(min_dist)
wf_limits_const = XYBoundaryConstraint(limits, 'convex_hull')
constraints = [turb_sep_constraint, wf_limits_const]

def setup_optimization(wfm, seed):

    # Initial coordinates for the optimization
    x_init, y_init = generate_random_array(n_tur=N_turbines, turbine=turbine, spacing=4, limits=limits, seed=seed)

    # Set up objective function
    if wfm == BQ_wfm or wfm == pseudo_BQ_wfm:
        def aep_func(x, y):
            return wfm.aep(x, y)
        
        def aep_gradient(x, y):
            return wfm.aep_gradients(x=x, y=y)
        
    elif wfm == flowers_wfm:
        def aep_func(x, y):
            return wfm.aep(x, y)
        
        def aep_gradient(x, y):
            return wfm.aep_gradient(x=x, y=y, gradient_method="Autograd")
        
        
    else:
        def aep_func(x, y):
            sim_res = wfm(x, y,
                    wd=site.default_wd,
                    ws=site.default_ws)
            return sim_res.aep().sum().values
        
        def aep_gradient(x, y):
            grad_x, grad_y = wfm.aep_gradients(x=x,
                                               y=y, 
                                               wd=site.default_wd,
                                               ws=site.default_ws)
            grad_x, grad_y = np.array([np.atleast_2d(grad_x), np.atleast_2d(grad_y)])
            return grad_x, grad_y

    # Set up cost model
    aep_comp = CostModelComponent(input_keys=[('x', x_init),('y', y_init)],
                                n_wt=N_turbines,
                                cost_function=aep_func,
                                cost_gradient_function=aep_gradient,
                                maximize=True,
                                objective=True,
                                output_keys=['AEP'])

    # Set up driver
    driver = EasyScipyOptimizeDriver(optimizer="SLSQP",
                                    maxiter=100,
                                    tol=1e-3)

    # Set up optimization problem
    topfarm_problem = TopFarmProblem(design_vars=dict(zip(['x', 'y'], [x_init, y_init])),
                                    cost_comp=aep_comp,
                                    driver=driver,
                                    constraints=constraints,
                                    n_wt=N_turbines,
                                    expected_cost=1e-5)

    return topfarm_problem

results_pywake = {}
results_bq_30 = {}
results_bq_50 = {}
results_bq_100 = {}
results_bq_150 = {}
results_bq_200 = {}
results_bq_360 = {}
results_flowers = {}
results_pseudo_BQ_50 = {}
results_pseudo_BQ_150 = {}
results_pseudo_BQ_360 = {}

for seed in range(10):
    print("\n" + "="*60)
    print(f"Running optimization with seed {seed}...")

    # PyWake optimization
    print("Optimizing with PyWake...")
    topfarm_problem = setup_optimization(pw_wfm, seed)
    start_time = time.time()
    _, state, recorder = topfarm_problem.optimize()
    end_time = time.time()
    final_aep = final_aep_function(state['x'], state['y'])
    results_pywake[seed] = {"AEP": final_aep, "Time": end_time - start_time}

    # BQ optimization
    bq_points = [30, 50, 100, 150, 200, 360]

    for points in bq_points:
        print(f"Optimizing with BQ with {points} points...")
        BQ_wfm.optimize_BQ_points(N_points=points, N_attempts=5)
        BQ_wfm.setup_gradients()
        topfarm_problem = setup_optimization(BQ_wfm, seed)
        start_time = time.time()
        _, state, recorder = topfarm_problem.optimize()
        end_time = time.time()
        final_aep = final_aep_function(state['x'], state['y'])
        globals()[f"results_bq_{points}"][seed] = {"AEP": final_aep, "Time": end_time - start_time}

    # FLOWERS optimization
    print("Optimizing with FLOWERS...")
    topfarm_problem = setup_optimization(flowers_wfm, seed)
    start_time = time.time()
    _, state, recorder = topfarm_problem.optimize()
    end_time = time.time()
    final_aep = final_aep_function(state['x'], state['y'])
    results_flowers[seed] = {"AEP": final_aep, "Time": end_time - start_time}

    MC_points = [50, 150, 360]

    # Pseudo-BQ optimization
    for points in MC_points:
        print(f"Optimizing with Pseudo-BQ with {points} MC points...")
        topfarm_problem = setup_optimization(pseudo_BQ_wfm, seed)
        pseudo_BQ_wfm.optimize_BQ_points(N_points=points, N_attempts=5)
        pseudo_BQ_wfm.setup_gradients(gradient_method=autograd)
        start_time = time.time()
        _, state, recorder = topfarm_problem.optimize()
        end_time = time.time()
        final_aep = final_aep_function(state['x'], state['y'])
        globals()[f"results_pseudo_BQ_{points}"][seed] = {"AEP": final_aep, "Time": end_time - start_time}

####################
# Create and print results table
####################

# Prepare data for table
table_data = []

# PyWake
aep_values = [results_pywake[seed]["AEP"] for seed in results_pywake.keys()]
time_values = [results_pywake[seed]["Time"] for seed in results_pywake.keys()]
table_data.append(["PyWake", np.median(aep_values), np.max(aep_values), np.median(time_values)])

# BQ variants
for points in [30, 50, 100, 150, 200, 360]:
    results_dict = globals()[f"results_bq_{points}"]
    aep_values = [results_dict[seed]["AEP"] for seed in results_dict.keys()]
    time_values = [results_dict[seed]["Time"] for seed in results_dict.keys()]
    table_data.append([f"BQ ({points})", np.median(aep_values), np.max(aep_values), np.median(time_values)])

results_dict = globals()[f"results_bq_360"]
aep_values = [results_dict[seed]["AEP"] for seed in results_dict.keys()]
time_values = [results_dict[seed]["Time"] for seed in results_dict.keys()]

# FLOWERS
aep_values = [results_flowers[seed]["AEP"] for seed in results_flowers.keys()]
time_values = [results_flowers[seed]["Time"] for seed in results_flowers.keys()]
table_data.append(["FLOWERS", np.median(aep_values), np.max(aep_values), np.median(time_values)])

# Pseudo-BQ
for points in MC_points:
    results_dict = globals()[f"results_pseudo_BQ_{points}"]
    aep_values = [results_dict[seed]["AEP"] for seed in results_dict.keys()]
    time_values = [results_dict[seed]["Time"] for seed in results_dict.keys()]
    table_data.append([f"Pseudo-BQ ({points})", np.median(aep_values), np.max(aep_values), np.median(time_values)])

# Create DataFrame and print
results_df = pd.DataFrame(table_data, columns=["WFM", "Median AEP", "Max AEP", "Median Time (s)"])
print("\n" + "="*60)
print("Optimization Results Summary")
print("="*60)
print(results_df.to_string(index=False))
print("="*60)


