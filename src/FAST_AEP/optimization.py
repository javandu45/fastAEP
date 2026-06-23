from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp

from FAST_AEP.utils import *

import time
import warnings
warnings.filterwarnings("ignore")

class optifast:

    def __init__(self, wind_farm, wind_farm_model, min_spacing=2, normalization=True, x_0=None, y_0=None, n_cpu=1, seed=None):

        self.wind_farm = wind_farm
        self.wind_farm_model = wind_farm_model
        self.min_spacing = min_spacing
        self.normalization = normalization
        self.n_cpu = n_cpu

        # Get wind farm data
        self.wf_data = get_wind_farm_data(wind_farm)
        self.wf_limits = get_limits(wind_farm)
        self.n_turbines = int(self.wf_data["turbine_count"].values[0])
        
        self.windTurbines = turbine_generator(wind_farm)

        # Generate initial positions
        if x_0 is None and y_0 is None:
            self.x_0, self.y_0 = generate_random_array(n_tur=self.n_turbines,
                                                turbine=self.windTurbines,
                                                spacing=2 if self.min_spacing is None else self.min_spacing,
                                                limits=self.wf_limits,
                                                seed=seed)
        else:
            self.x_0 = x_0
            self.y_0 = y_0
        
        # Get minimum x and y to center coordinates if necessary
        self.min_x = self.wf_limits[:, 0].min()
        self.min_y = self.wf_limits[:, 1].min()

        if x_0 is None and y_0 is None:

            # Avoid double centering if initial positions are already centered
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
            self.aep_0 = self.wind_farm_model.aep(self.x_0, self.y_0)

        else:
            self.aep_0 = 1


    def _setup_normalization(self):
        """Normalize coorindates with respect to maximum coordinates in the wind farm"""

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
        
        # The normalization is applied using the grid layout component, an intermediate component
        normalization_component = CostModelComponent(input_keys=[('x_norm', self.x_0_norm), ('y_norm', self.y_0_norm)],
                                                    n_wt=len(self.x_0),
                                                    cost_function=normalize_coords,
                                                    cost_gradient_function=normalize_coords_grad,
                                                    output_keys=[('x', self.x_0), ('y', self.y_0)],
                                                    objective=False,
                                                    use_constraint_violation=False)
        
        return normalization_component


    def _setup_objective_function(self):
        """Set up objective function, in this case AEP"""

        def aep_function(x, y):
            # Some wind farm models (e.g. Flowers) do not accept n_cpu; ignore it for those models
            model_name = self.wind_farm_model.__class__.__name__.lower()
            module_name = getattr(self.wind_farm_model.__class__, '__module__', '').lower()
            if 'flowers' in model_name or 'flowers' in module_name:
                aep = self.wind_farm_model.aep(x, y)
            else:
                aep = self.wind_farm_model.aep(x, y, n_cpu=self.n_cpu)
            return aep/self.aep_0
        
        return aep_function
    

    def _setup_gradient_function(self):
        """Set up gradient function, in this case AEP gradient"""

        def aep_gradient(x, y):
            model_name = self.wind_farm_model.__class__.__name__.lower()
            module_name = getattr(self.wind_farm_model.__class__, '__module__', '').lower()
            if 'flowers' in model_name or 'flowers' in module_name:
                grad_x, grad_y = self.wind_farm_model.aep_gradient(x, y)
            else:
                grad_x, grad_y = self.wind_farm_model.aep_gradient(x, y, n_cpu=self.n_cpu)
            return grad_x/self.aep_0, grad_y/self.aep_0
        
        return aep_gradient
    

    def _setup_constraints(self):
        """Set up constraints"""

        # Boundary constraints
        wf_limits_const = XYBoundaryConstraint(self.wf_limits, 'polygon')

        # Distance constraints
        if self.min_spacing is not None:
            minimum_distance = self.min_spacing * self.windTurbines.diameter()
            turbine_separation_constrain = SpacingConstraint(min_spacing=minimum_distance)

            return [turbine_separation_constrain, wf_limits_const]
        
        else:
            # For the double step optimization, the first optimization does not consider distance constraints
            return [wf_limits_const]
        

    def setup_problem(self, tolerance=1, expected_cost=1, max_iter=100):
        """Set up the TopFarm optimization problem"""

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

        # Normalize if necessary
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
        
        return topfarm_problem

        