# %%

# Import fast AEP
from FAST_AEP.BQ import bayesian_quadrature
from FAST_AEP.FLOWERS import NOJ_flowers, TurbOPark_flowers

import warnings
warnings.filterwarnings("ignore")

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.literature.noj import Jensen_1983
import numpy as np
from FAST_AEP.utils import generate_random_array, generic_site, turbine_generator, get_limits
from matplotlib import pyplot as plt
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp
from py_wake.utils.gradients import autograd, fd
import time

# Set up problem conditions
site = generic_site("Hornsea_Project_2_-_Phase_1_Breesea")
# site = Hornsrev1Site()
turbine = turbine_generator("Hornsea_Project_2_-_Phase_1_Breesea")
N_turbines = 9
turbines_per_row = int(np.sqrt(N_turbines))
sep = turbines_per_row*6*turbine.diameter()
limits = np.array([(0, 0), (sep, 0), (sep,sep), (0, sep)])
limits = get_limits("Hornsea_Project_2_-_Phase_1_Breesea")
min_dist = 5*turbine.diameter()
x_init, y_init = generate_random_array(n_tur=N_turbines, turbine=turbine, spacing=4, limits=limits)
# x_init, y_init = Hornsrev1Site().initial_position.T

# Set up wind farm model
wfm = Jensen_1983(site=site, windTurbines=turbine, k=0.04)
# wfm = Nygaard_2022(site=site, windTurbines=turbine)

sim_res = wfm(x_init, y_init,
            wd=site.default_wd,
            ws=site.default_ws)

initial_aep = sim_res.aep().sum().values

print(f"Initial AEP from flow model: {initial_aep:.2f}")

BQ_wfm = bayesian_quadrature(site=site,
                            windTurbines=turbine,
                            flow_model=wfm,
                            x0=x_init,
                            y0=y_init,
                            N_train=3000,
                            N_MC=4000, 
                            aep_method="BQ",
                            kernel_type="linear")

BQ_wfm.train_and_get_kernel()

# %%

BQ_wfm.optimize_BQ_points(N_points=360, N_attempts=10, tol=1e-8, jitter=0.1)
BQ_wfm.plot_optimized_points()

print("AEP from BQ: ", BQ_wfm.aep(x_init, y_init))
# BQ_wfm.setup_gradients(gradient_method=autograd)

# # # %%
# time_start = time.time()
# grad_x, grad_y = BQ_wfm.aep_gradients(x=x_init, y=y_init)
# time_end = time.time()
# print(f"Time taken to compute gradients: {time_end - time_start:.2f} seconds")

# BQ_wfm.setup_gradients(gradient_method=fd)

# # # %%
# time_start = time.time()
# grad_x_fd, grad_y_fd = wfm.aep_gradients(x=x_init, y=y_init)
# time_end = time.time()
# print(f"Time taken to compute gradients: {time_end - time_start:.2f} seconds")


#%% Set up FLOWERS AEP
# wfm_flowers = turbo_flowers_claude(site=site, windTurbines=turbine, n_terms=10)
wfm_flowers = NOJ_flowers(site=site, windTurbines=turbine, n_terms=10)
# wfm_flowers = TurbOPark_flowers(site=site, windTurbines=turbine, n_terms=10)


# %%

BQ_wfm.setup_gradients(gradient_method=autograd)

# Set up objective function
def aep_func(x, y):

    fun = BQ_wfm.aep(x, y)
    # fun = wfm_flowers.aep(x, y)
    # sim_res = wfm(x, y,
    #         wd=site.default_wd,
    #         ws=site.default_ws)
    
    # fun = sim_res.aep().sum().values

    return fun

# Set up gradients
def aep_gradient(x, y):
    grad_x, grad_y = BQ_wfm.aep_gradients(x=x, y=y)
    # grad_x, grad_y = wfm_flowers.aep_gradient(gradient_method="Autograd", x=x, y=y)
    # grad_x, grad_y = wfm.aep_gradients(gradient_method=autograd,
    #                                    wrt_arg=["x", "y"],
    #                                    x=x,
    #                                    y=y, 
    #                                    wd=site.default_wd,
    #                                    ws=site.default_ws,
    #                                    normalize_probabilities=True)
    # grad_x, grad_y = np.array([np.atleast_2d(grad_x), np.atleast_2d(grad_y)])

    return grad_x, grad_y

min_dist = 4*turbine.diameter()

# Set up constraints
turb_sep_constraint = SpacingConstraint(min_dist)
wf_limits_const = XYBoundaryConstraint(limits, 'convex_hull')
constraints = [turb_sep_constraint, wf_limits_const]

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
                                 expected_cost=1e-6,
                                 plot_comp=XYPlotComp())


# Run optimization
time_start = time.time()
print("Starting optimization...")
_, state, recorder = topfarm_problem.optimize()
time_end = time.time()

x_opt = state["x"]
y_opt = state["y"]

sim_res = wfm(x_opt, y_opt,
              wd=site.default_wd,
              ws=site.default_ws)

aep_final = sim_res.aep().sum().values

print(f"Optimization completed in {time_end - time_start:.2f} seconds")
print(f"Final AEP from flow model: {aep_final:.2f}")

cost_history = recorder.get("cost")

plt.figure(figsize=(10, 5))
plt.plot(-cost_history, marker="o")
plt.title("AEP Optimization History")
plt.xlabel("Iteration")
plt.ylabel("AEP (GWh)")
plt.grid()
plt.show()

final_aep_BQ = BQ_wfm.aep(x_opt, y_opt)
print(f"Final AEP from BQ surrogate: {final_aep_BQ:.2f}")


# %%
