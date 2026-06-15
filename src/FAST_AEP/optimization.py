from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp

from FAST_AEP.utils import *

import time


class optifast:

    def __init__(self, wind_farm, wind_farm_model, min_spacing=2, normalization=True):

        self.wind_farm = wind_farm
        self.wind_farm_model = wind_farm_model
        self.min_spacing = min_spacing
        self.normalization = normalization

        # Get wind farm data
        self.wf_data = get_wind_farm_data(wind_farm)
        self.wf_limits = get_limits(wind_farm)
        self.n_turbines = int(self.wf_data["turbine_count"].values[0])
        
        self.windTurbines = turbine_generator(wind_farm)

        # Generate initial positions
        self.x_0, self.y_0 = generate_random_array(n_tur=self.n_turbines,
                                            turbine=self.windTurbines,
                                            spacing=self.min_spacing,
                                            limits=self.wf_limits,
                                            seed=None)
        
        # Center coordinates if necessary
        self.min_x = self.wf_limits[:, 0].min()
        self.min_y = self.wf_limits[:, 1].min()

        self.x_0 = self.x_0 - self.min_x
        self.y_0 = self.y_0 - self.min_y

        # Center limits
        limits_centered = np.zeros_like(self.wf_limits)

        limits_centered[:, 0] = (self.wf_limits[:, 0] - self.min_x)
        limits_centered[:, 1] = (self.wf_limits[:, 1] - self.min_y)

        self.wf_limits = limits_centered

        # Obtain max x and y to normalize coordinates if necessary
        self.max_x = self.wf_limits[:, 0].max()
        self.max_y = self.wf_limits[:, 1].max()

        if self.normalization:
            time_i = time.time()
            self.aep_0 = self.wind_farm_model.aep(self.x_0, self.y_0)
            time_f = time.time()
            print(f"Initial AEP calculated in {time_f - time_i:.2f} seconds")

        else:
            self.aep_0 = 1


    def _setup_normalization(self):

        def normalize_coords(x_norm, y_norm):

            x_real = x_norm * self.max_x
            y_real = y_norm * self.max_y

            return [x_real, y_real]
        
        def normalize_coords_grad(x_norm, y_norm): 

            dx_dxnorm = np.eye(len(x_norm)) * self.max_x
            dy_dynorm = np.eye(len(y_norm)) * self.max_y
            dx_dynorm = np.zeros((len(x_norm), len(y_norm)))
            dy_dxnorm = np.zeros((len(y_norm), len(x_norm)))
            
            return [[dx_dxnorm, dy_dxnorm], [dx_dynorm, dy_dynorm]]
        
        normalization_component = CostModelComponent(input_keys=[('x_norm', self.x_0_norm), ('y_norm', self.y_0_norm)],
                                                    n_wt=len(self.x_0),
                                                    cost_function=normalize_coords,
                                                    cost_gradient_function=normalize_coords_grad,
                                                    output_keys=[('x', self.x_0), ('y', self.y_0)],
                                                    objective=False,
                                                    use_constraint_violation=False)
        
        return normalization_component


    def _setup_objective_function(self):

        def aep_function(x, y):
            aep = self.wind_farm_model.aep(x, y)
            return aep/self.aep_0
        
        return aep_function
    

    def _setup_gradient_function(self):

        def aep_gradient(x, y):
            grad_x, grad_y = self.wind_farm_model.aep_gradient(x, y)
            return grad_x/self.aep_0, grad_y/self.aep_0
        
        return aep_gradient
    

    def _setup_constraints(self):

        # Distance constraints
        minimum_distance = self.min_spacing * self.windTurbines.diameter()
        turbine_separation_constrain = SpacingConstraint(min_spacing=minimum_distance)

        # Boundary constraints
        wf_limits_const = XYBoundaryConstraint(self.wf_limits, 'polygon')

        # return [turbine_separation_constrain, wf_limits_const]
        return wf_limits_const
        

    def setup_problem(self, tolerance=1, expected_cost=1, max_iter=100, seed=None):

        # Generate functions and constraints
        aep_function = self._setup_objective_function()
        aep_gradient = self._setup_gradient_function()
        constraints = self._setup_constraints()

        # Set up cost component
        cost_component = CostModelComponent(input_keys=[('x', self.x_0),('y', self.y_0)],
                                            n_wt=self.n_turbines,
                                            cost_function=aep_function,
                                            cost_gradient_function=aep_gradient,
                                            maximize=True,
                                            objective=True,
                                            output_keys=['AEP'])
        
        # Set up driver
        driver = EasyScipyOptimizeDriver(optimizer="SLSQP",
                                        maxiter=max_iter,
                                        tol=tolerance)

        if self.normalization:

            self.x_0_norm = self.x_0 / self.max_x
            self.y_0_norm = self.y_0 / self.max_y

            design_vars = dict(zip(['x_norm', 'y_norm'], [(self.x_0_norm), (self.y_0_norm)]))
            normalization_func = self._setup_normalization()

        else:
            design_vars = dict(zip(['x', 'y'], [self.x_0, self.y_0]))
            normalization_func = None

        # Set up optimization problem
        topfarm_problem = TopFarmProblem(design_vars=design_vars,
                                        cost_comp=cost_component,
                                        driver=driver,
                                        constraints=constraints,
                                        grid_layout_comp = normalization_func,
                                        n_wt=self.n_turbines,
                                        expected_cost=expected_cost)
                                        # plot_comp = XYPlotComp())
        
        return topfarm_problem
                                            
    