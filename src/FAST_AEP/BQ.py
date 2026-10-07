# from examples.utils import *
import numpy as np
import gpytorch
import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from py_wake.utils.gradients import autograd, fd
import autograd.numpy as anp

class bayesian_quadrature():

    """
    Bayesian quadrature implementation for AEP computation

    This algorithm uses a Gaussian process surrogate to model an approximation of the power function, which then using
    bayesian quadrature computes the AEP by integrating. The points used for the surrogate model are optimized in order
    to minimize the uncertainty in the AEP estimation. The gaussian process surrogate uses a kernel function, which needs
    to be trained with a set of points sampled from the wind rose distribution. Once the integration points have been 
    optimized, they can be used to compute AEP for any layout, provided that the wind rose remains constant.

    Based on "A Probabilistic Approach to Estimating Wind Farm Annual Energy Production with Bayesian Quadrature"
    https://doi.org/10.2514/6.2020-1951
    """

    def __init__(self, site, windTurbines, flow_model, x0, y0, N_train=2000, N_MC=5000, aep_method="BQ", kernel_type="Periodic"):

        """
        Model initialization.
        
        Parameters
        ----------
        Site : Site
            Site Object containing the wind rose information and characteristics of the location
        windTurbine : windTurbines
            windTurbines object representing the wake generating wind turbines
        flow_model : callable
            Py_wake wind farm flow model, which will be used to evaluate the power function at the different atmospheric
            conditions
        x0 : array_like
            Initial x coordinates of the turbines, used for training the GP surrogate model 
        y0 : array_like
            Initial y coordinates of the turbines, used for training the GP surrogate model
        N_train : int, optional
            Number of training points used to train the GP surrogate model, default is 2000
        N_MC : int, optional
            Number of Monte Carlo samples used to compute the weights for the Bayesian quadrature, default is 5000
        aep_method : str, optional
            Method to compute the AEP, either "BQ" for Bayesian quadrature or "RQ" for Rectangular Quadrature integration, default is "BQ". 
            If "BQ", the AEP will be computed using the optimized points and the kernel function, while if "RQ", the AEP will be
            computed using the Monte Carlo samples and their corresponding probabilities.
        kernel_type : str, optional
            Type of kernel to use in the GP surrogate model, default is "Periodic". If "Periodic", the kernel will be
            defined in a way that accounts for the periodicity of the wind direction, if "Standard", the kernel will be
            defined in a standard way without accounting for periodicity.
        """

        self.site = site
        self.windTurbines = windTurbines
        self.flow_model = flow_model
        self.x0 = x0
        self.y0 = y0
        self.N_train = N_train
        self.kernel_type = kernel_type
        self.aep_method = aep_method
        self.name = aep_method

        # Wind speed and direction characteristics from the site
        self.A_weibull = site.ds.Weibull_A.values
        self.k_weibull = site.ds.Weibull_k.values
        self.p_direction = site.ds.Sector_frequency.values[:-1]/sum(site.ds.Sector_frequency.values[:-1])
        self.wds = np.deg2rad(site.ds.wd.values[:-1])

        # Parameters for Monte Carlo integration in weights computation
        self.N_MC = N_MC
        self.X_mc = self._X_sampling(self.N_MC)
        p_mc = self._X_probabilities(self.X_mc)
        self.p_mc = p_mc / np.sum(p_mc)

        # Preconvert RQ samples to torch for faster optimization with torch
        self.X_mc_torch = torch.tensor(self.X_mc, dtype=torch.float64)
        self.p_mc_torch = torch.tensor(self.p_mc, dtype=torch.float32)

   
    def _X_sampling(self, N, seed=None):

        """
        Sample N points from the wind rose distribution, with upper and lower limits corresponging to normal wind 
        turbine operating conditions.

        Parameters
        ----------
        N : int
            Number of points to sample

        Returns
        -------
        X : ndarray
            Array of shape (N, 3) if kernel_type is "Periodic", or (N, 2) if kernel_type is "Standard", containing the
            sampled points. The first column corresponds to wind speed, the second column corresponds to the cosine of
            the wind direction and the third column corresponds to the sine of the wind direction if kernel_type is "Periodic",
            while for "Standard" the second column corresponds to the wind direction in radians.
        seed : int, optional
            Random seed for reproducibility, default is None (no seed).
        """

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Sample wind directions based on the wind rose distribution
        wd = rng.choice(self.wds, size=N, p=self.p_direction)

        # Find the closest standard angle
        angle_diff = np.abs(self.wds[:, np.newaxis] - wd)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        index = np.argmin(angle_diff, axis=0)

        ws = np.empty(N)
        for i in range(N):
            scale = self.A_weibull[index[i]]
            shape = self.k_weibull[index[i]]

            # Make sure values are withing operational ranges
            while True:
                w = scale * rng.weibull(shape)
                if 3 <= w <= 25:
                    ws[i] = w
                    break

        # Convert to cos and sin if kernel is periodic
        if self.kernel_type == "Periodic":
            coswd = np.cos(wd)
            sinwd = np.sin(wd)
            return np.column_stack([ws, coswd, sinwd])

        return np.column_stack([ws, wd])    


    def _X_probabilities(self, X):

        """
        Obtain the joint probability of wind speed and direction for an array of points, based on the wind rose distribution
        of the site.

        Parameters
        ----------
        X : ndarray
            Array of shape (N, 3) if kernel_type is "Periodic", or (N, 2) if kernel_type is "Standard", containing the
            points for which to compute the probabilities. The first column corresponds to wind speed, the second column 
            corresponds to the cosine of the wind direction and the third column corresponds to the sine of the wind direction
            if kernel_type is "Periodic", while for "Standard" the second column corresponds to the wind direction in radians.

        Returns
        -------
        probabilities : ndarray
            Array of shape (N,) containing the joint probabilities of wind speed and direction for each point in X.
        """

        if self.kernel_type == "Periodic":
            ws = X[:, 0]
            coswd = X[:, 1]
            sinwd = X[:, 2]

            # Wind direction is needed to determine its probability and closest angle in the wind rose
            wd = np.arctan2(sinwd, coswd) % (2*np.pi)

        else:
            ws = X[:, 0]
            wd = X[:, 1]

        # Find the closest standard angle
        angle_diff = np.abs(self.wds[:, np.newaxis] - wd)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        index = np.argmin(angle_diff, axis=0)

        # Probability of the wind speed using a weibull distribution
        p_ws = self.k_weibull[index] / self.A_weibull[index] * (ws / self.A_weibull[index])**(self.k_weibull[index]-1) * np.exp(-(ws/self.A_weibull[index])**self.k_weibull[index])
        p_wd = self.p_direction[index]

        return p_ws*p_wd
    
    
    def _convert_X(self, X):
        """
        Convert the input array X to wind speed and direction, depending on the kernel type.
        """

        if self.kernel_type == "Periodic":
            ws = X[:, 0]
            coswd = X[:, 1]
            sinwd = X[:, 2]
            wd = np.arctan2(sinwd, coswd) % (2*np.pi)
            wd = np.rad2deg(wd) 
            return ws, wd
        else:
            ws = X[:, 0]
            wd = X[:, 1]
            wd = np.rad2deg(wd)
            return ws, wd
        

    def _power_function(self, x, y, X, n_cpu=1):

        """
        It returns the wind farm power for an array of points with different wind speed and direction conditions,
        evaluated using the flow model.
        
        Parameters
        ----------
        x : array_like
            x coordinates of the turbines, used for evaluating the power function
        y : array_like
            y coordinates of the turbines, used for evaluating the power function
        X : ndarray
            Array of shape (N, 3) if kernel_type is "Periodic", or (N, 2) if kernel_type is "Standard", containing the
            points for which to compute the power function. The first column corresponds to wind speed, the second column
            corresponds to the cosine of the wind direction and the third column corresponds to the sine of the wind
            direction if kernel_type is "Periodic".
        n_cpu : int, optional
            Number of CPU cores to use for parallel evaluation (default is 1).
        """

        # Convert X to wind speed and direction
        ws, wd = self._convert_X(X)

        time_stamp = np.arange(len(ws))

        sim_res = self.flow_model(x, y,
                                wd=wd,
                                ws=ws,
                                time=time_stamp,
                                n_cpu=n_cpu)

        power_vector = np.sum(np.array(sim_res.Power), axis=0)

        return power_vector

    
    def _train_GP(self, N_train):

        """
        Train the Gaussian process surrogate model using N_train points sampled from the wind rose distribution.
        The kernel lengthscales are obtained during the training process and stored for later use in the kernel
        function.

        Parameters
        ----------
        N_train : int
            Number of training points used to train the GP surrogate model, recommended to be at least 1000)

        """

        print("Obtaining legnthscales for Kernel...")

        # Obtain training points and corresponding power values
        X_train = self._X_sampling(N_train)
        y_train = self._power_function(self.x0, self.y0, X_train)

        train_X = torch.tensor(X_train, dtype=torch.float32)

        y_mean = np.mean(y_train)
        y_std = np.std(y_train)

        # Store normalization parameters for denormalization during prediction
        self._y_mean_train = y_mean
        self._y_std_train = y_std

        y_train_norm = (y_train - y_mean) / y_std
        train_y = torch.tensor(y_train_norm, dtype=torch.float32)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Train the GP surrogate model and obtain the trained kernel
        self.model = WindFarmGP(train_X, train_y, self.likelihood, n_dims=X_train.shape[1])
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(150):

            optimizer.zero_grad()
            output = self.model(train_X)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Extract lengthscales from the trained kernel and store them for later use in the kernel function
        lengthscales = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
        self.lengthscales = lengthscales

        print(f"Trained lengthscales: {lengthscales}")

    
    def train_and_get_kernel(self):

        """
        Train the Gaussian process surrogate model and store the trained kernel for later use in the kernel function.
        """

        # Train GP
        self._train_GP(N_train=self.N_train)

        # Put model in evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Store trained kernel directly from GPyTorch
        self.kernel_trained = self.model.covar_module.base_kernel


    def _kernel(self, X1, X2):

        """
        Compute the kernel matrix between two sets of points X1 and X2 using the trained kernel from the Gaussian
        process surrogate model.
        
        Parameters
        ----------
        X1 : ndarray
            Array of shape (N1, 3) if kernel_type is "Periodic", or (N1, 2) if kernel_type is "Standard", containing
            the first set of points. The first column corresponds to wind speed, the second column corresponds to the
            cosine of the wind direction and the third column corresponds to the sine of the wind direction if
            kernel_type is "Periodic", while for "Standard" the second column corresponds to the wind direction
            in radians.
        X2 : ndarray
            Array, similar to X1, but containing the second set of points.

        Returns
        -------
        K : ndarray
            Kernel matrix of shape (N1, N2) containing the kernel values between each pair of points in X1 and X2.
        """

        X1_torch = X1.detach().clone().to(dtype=torch.float64) if isinstance(X1, torch.Tensor) else torch.tensor(X1, dtype=torch.float64)
        X2_torch = X2.detach().clone().to(dtype=torch.float64) if isinstance(X2, torch.Tensor) else torch.tensor(X2, dtype=torch.float64)

        with torch.no_grad():
            K = self.kernel_trained(X1_torch, X2_torch).evaluate()

        return K


    def _get_weights(self, X):

        """
        Compute the weights for the Bayesian quadrature using the kernel function and the probabilities of the Monte Carlo
        samples.

        Parameters
        ----------
        X : ndarray
            Array of shape (N, 3) if kernel_type is "Periodic", or (N, 2) if kernel_type is "Standard", containing the
            points for which to compute the weights. The first column corresponds to wind speed, the second column corresponds
            to the cosine of the wind direction and the third column corresponds to the sine of the wind direction if
            kernel_type is "Periodic", while for "Standard" the second column corresponds to the wind direction in radians.

        Returns
        -------
        w : torch.tensor
            Tensor of shape (N,) containing the weights for each point in X.
        """

        K_mc = self.kernel_trained(self.X_mc_torch, X).evaluate()
        w = torch.sum(K_mc * self.p_mc_torch[:, None], dim=0)

        return w    
    

    def optimize_BQ_points(self, N_points, N_attempts=5, x0=None, tol=1e-8, jitter=0.1):

        """
        Optimize the points used for the surrogate model in order to minimize the variance in the AEP estimation.
        The optimization is performed using the L-BFGS-B algorithm from scipy, with gradients computed using PyTorch autograd.
        
        Parameters
        ----------
        N_points : int
            Number of points to optimize. More points results in a better estimation of the AEP, but also increases the
            computational cost of the optimization, and AEP computation.
        N_attempts : int, optional
            Number of optimization attempts with different random initial points, default is 5. The solution resulting in the
            lowest variance in the AEP estimation will be selected as the final solution.
        x0 : ndarray, optional
            Initial guess for the optimized points.
        tol : float, optional
            Tolerance for the optimization.
        jitter : float, optional
            Jitter to add to the diagonal of the kernel matrix for numerical stability during optimization.
        """

        variance_list = []
        X_list = []

        # If initial points are given, avoid multiple optimization attempts
        if x0 is not None:
            if N_attempts > 1:
                print("Warning: x0 provided but N_attempts > 1, setting N_attempts to 1 to avoid multiple optimizations with the same initial points.")
                N_attempts = 1

        # Run multiple random initializations to avoid local minima, and select the most optimal one
        for i in range(N_attempts):

            # If coordinates are not given, set the initial points to random samples from the wind rose distribution
            if x0 is None:
                x0 = self._X_sampling(N_points, seed=None)

            if self.kernel_type == "Periodic":
                
                # Initialize normalized initial points for optimization
                x0_norm = np.zeros((N_points, 3))
                x0_norm[:,0] = x0[:,0] / 25
                x0_norm[:,1] = x0[:,1]
                x0_norm[:,2] = x0[:,2]
                x0_norm = x0_norm.flatten()

                # Bounds for (ws_norm, cos_theta, sin_theta)
                bounds_periodic = []
                for _ in range(N_points):
                    bounds_periodic.append((3/25, 1))   # wind speed
                    bounds_periodic.append((-1, 1))     # cos(theta)
                    bounds_periodic.append((-1, 1))     # sin(theta)

                def objective_torch(db):

                    db_np = db.reshape(-1, 3)
                    
                    # Project cos/sin onto unit circle before passing to torch
                    # This enforces periodicity without an explicit constraint
                    ws_np = db_np[:, 0]
                    cos_np = db_np[:, 1]
                    sin_np = db_np[:, 2]
                    norm = np.sqrt(cos_np**2 + sin_np**2 + 1e-8)
                    cos_np = cos_np / norm
                    sin_np = sin_np / norm

                    db_projected = np.stack([ws_np, cos_np, sin_np], axis=1)
                    db_t = torch.tensor(db_projected, dtype=torch.float64, requires_grad=True)

                    ws = db_t[:,0] * 25
                    X = torch.stack([ws, db_t[:,1], db_t[:,2]], dim=1)

                    # Obtain weights
                    w = self._get_weights(X)

                    # Compute objective function
                    K = self.kernel_trained(X, X).evaluate()
                    K += jitter * torch.eye(N_points, dtype=torch.float64)
                    L = torch.linalg.cholesky(K)
                    Kinv_w = torch.cholesky_solve(w[:,None], L).squeeze()
                    fun = -torch.dot(w, Kinv_w)

                    # Obtain gradients using torch
                    fun.backward()
                    grad = db_t.grad.detach().numpy().flatten()

                    return (fun.item(), grad)

                result = minimize(objective_torch,
                                x0_norm,
                                method='L-BFGS-B',
                                tol=tol,
                                options={"maxiter": 10000, "maxfun": 100000},
                                bounds=bounds_periodic,
                                jac=True)

                X_opt = result.x.reshape(N_points, 3)

                solution = np.zeros((N_points, 3))
                solution[:,0] = X_opt[:,0] * 25
                # Re-normalize cos/sin onto unit circle for final solution
                norm = np.sqrt(X_opt[:,1]**2 + X_opt[:,2]**2 + 1e-8)
                solution[:,1] = X_opt[:,1] / norm
                solution[:,2] = X_opt[:,2] / norm

            else:
                
                # Establish normalized bounds
                bounds = []
                for _ in range(N_points):
                    bounds.append((3/25, 1))        
                    bounds.append((0, 1))

                # Initialize normalized initial points for optimization
                x0_norm = np.zeros((N_points, 2))
                x0_norm[:,0] = x0[:,0] / 25
                x0_norm[:,1] = x0[:,1] / (2*np.pi)
                x0_norm = x0_norm.flatten()

                def objective_torch(db):

                    db = torch.tensor(db.reshape(-1, 2), dtype=torch.float64, requires_grad=True)

                    ws = db[:,0] * 25
                    theta = db[:,1] * 2*np.pi
                    X = torch.stack([ws, theta], dim=1)

                    # Obtain weights
                    w = self._get_weights(X)

                    # Compute objective function
                    K = self.kernel_trained(X, X).evaluate()
                    K += jitter * torch.eye(N_points, dtype=torch.float64)
                    L = torch.linalg.cholesky(K)
                    Kinv_w = torch.cholesky_solve(w[:,None], L).squeeze()
                    fun = -torch.dot(w, Kinv_w)

                    fun.backward()
                    grad = db.grad.detach().numpy().flatten()

                    return (fun.item(), grad)

                result = minimize(objective_torch,
                                x0_norm,
                                method='L-BFGS-B',
                                tol=tol,
                                options={"maxiter": 10000, "maxfun": 100000},
                                bounds=bounds,
                                jac=True)

                X_opt = result.x.reshape(N_points, 2)

                solution = np.zeros((N_points, 2))
                solution[:,0] = X_opt[:,0] * 25
                solution[:,1] = X_opt[:,1] * 2*np.pi

            variance_list.append(-result.fun)

            # print(f"Attempt {i+1}/{N_attempts}, Variance: {-result.fun:.10f}")

            X_list.append(solution)

            # Set x0 to none after iteration so that new samples are obtained
            x0 = None

        # Select the solution with the lowest variance (max result)
        best_idx = np.argmax(variance_list)
        # print(f"Best variance: {variance_list[best_idx]:.10f} at attempt {best_idx+1}")
        self.optimized_points = X_list[best_idx]

        # Leave weights and kernel computed for faster AEP computation
        self.w_opt = self._get_weights(torch.tensor(self.optimized_points, dtype=torch.float64)).detach().numpy()
        self.k_opt = self._kernel(self.optimized_points, self.optimized_points).detach().numpy()    


    def aep(self, x, y, n_cpu=1):

        """
        Compute the AEP using the optimized points for the surrogate model and the kernel function.

        Parameters
        ----------
        x : array_like
            x coordinates of the turbines, used for evaluating the power function
        y : array_like
            y coordinates of the turbines, used for evaluating the power function
            
        Returns
        -------
        AEP : float
            Annual Energy Production, in GWh
        """

        X = self.optimized_points

        if self.aep_method == "BQ":        

            X = torch.tensor(X, dtype=torch.float64)

            # Used saved weight and kernerl
            w_ = self.w_opt
            k_ = self.k_opt

            # Obtain power vector for the optimized points
            y_ = self._power_function(x, y, X, n_cpu=n_cpu)

            # Perform bayesian quadrature to obtain AEP
            AEP = 8760/1e9 * w_.T @ np.linalg.inv(k_) @ y_

        elif self.aep_method == "RQ":

            # Obtain power vector for the Monte Carlo samples
            power_vector = self._power_function(x, y, X, n_cpu=n_cpu)
            freqs = self._X_probabilities(X)
            frequency_normalized = freqs / np.sum(freqs)

            # Compute AEP using Monte Carlo integration
            AEP = np.sum(power_vector * frequency_normalized) * 8760 / 1e9

        return AEP
    

    def setup_gradients(self, gradient_method=autograd, n_cpu=1):

        """
        Set up AEP gradients based on the optimized points and the method selected for AEP computation

        Parameters
        ----------
        gradient_method : callable, optional
            Method to compute the gradients. Use autograd of fd from py_wake.utils.gradients.
        """

        # Convert optimized points to wind speed and direction for power function evaluation
        ws, wd = self._convert_X(self.optimized_points)  

        if self.aep_method == "BQ":

            c_const = (8760 / 1e9 * self.w_opt.T @ np.linalg.inv(self.k_opt)).ravel()

            # Define a function that computes the weighted power sum for the optimized points, which is the core of the AEP computation.
            def wf_power(x, y):

                power_i = self.flow_model(x=x, y=y, ws=ws, wd=wd, return_simulationResult=False, time=True)[2]
                power_wf = anp.sum(power_i, 0)   # shape: (len(wd),)

                return anp.dot(c_const, power_wf)

            grad_fun = gradient_method(wf_power, vector_interdependence=True, argnum=[0,1])

            # Gradient function
            def _aep_gradient(x, y):

                daep_dx, daep_dy = grad_fun(x, y)

                return daep_dx, daep_dy
            
            # Save the gradient function as an attribute of the class for later use in optimization
            self.aep_gradients_function = _aep_gradient

        elif self.aep_method == "RQ":

            # Define the gradient functions using PyWake
            def _aep_gradient(x, y):

                jx, jy = self.flow_model.aep_gradients(gradient_method=gradient_method,
                                                        wrt_arg=["x", "y"],
                                                        x=x,
                                                        y=y,
                                                        wd=wd,
                                                        ws=ws,
                                                        time=True,
                                                        n_cpu=n_cpu)
            
                daep = np.array([np.atleast_2d(jx), np.atleast_2d(jy)])
                daep_dx, daep_dy = daep

                return daep_dx, daep_dy

            # Save the gradient function as an attribute of the class for later use in optimization
            self.aep_gradients_function = _aep_gradient

        return 


    def aep_gradient(self, x=None, y=None):

        """
        Compute the gradient of the AEP with respect to the turbine coordinates using automatic diffierentiation
        or finite differences, as determine in the setup_gradients function.

        Parameters
        ----------
        x : array_like, optional
            x coordinates of the turbines, used for evaluating the power function, required if gradient_method is "FD"
        y : array_like, optional
            y coordinates of the turbines, used for evaluating the power function, required if gradient_method is "FD"
        X : ndarray, optional
            Array of shape (N, 3) if kernel_type is "Periodic", or (N, 2) if kernel_type is "Standard". If not given,
            the gradients will be computed for the optimized points. 

        Returns
        -------
        dy_dx : ndarray
            Array of shape (N, n_turbines) containing the gradient of the AEP with respect to the x coordinates of the
            turbines.
        dy_dy : ndarray
            Array of shape (N, n_turbines) containing the gradient of the AEP with respect to the y coordinates of the
            turbines.
        """

        dy_dx, dy_dy = self.aep_gradients_function(x, y)

        return dy_dx, dy_dy


    def predict_power(self, X, return_std=False):

        """
        Predict the power for an array of points using the Gaussian process surrogate model.

        Parameters
        ----------
        X : ndarray
            Array of shape (N, 3) if kernel_type is "Periodic", or (N, 2) if kernel_type is "Standard", containing the
            points for which to compute the power function.
        return_std : bool, optional
            Whether to return the standard deviation of the predictions, default is False.

        Returns
        -------
        mean_power : ndarray
            Array of shape (N,) containing the predicted power for each point in X.
        std_power : ndarray, optional
            Array of shape (N,) containing the standard deviation of the predictions for each point in X, returned only
            if return_std is True.
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Prepare X based on input format and kernel type
        if X.shape[1] == 2:
            if self.kernel_type == "Periodic":
                ws = X[:, 0]
                wd = X[:, 1]
                X_prepared = np.column_stack([ws, np.cos(wd), np.sin(wd)])
            else:
                X_prepared = X
        elif X.shape[1] == 3:
            X_prepared = X
        
        # Convert to torch tensor
        X_torch = torch.tensor(X_prepared, dtype=torch.float32)
        
        # Get predictions from GP with no gradient computation
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.eval_cg_tolerance(1e-2):
            predictions = self.likelihood(self.model(X_torch))
            mean_norm = predictions.mean.numpy()
            std_norm = predictions.stddev.numpy()
        
        # Denormalize predictions to original power scale
        if hasattr(self, '_y_mean_train') and hasattr(self, '_y_std_train'):
            mean_power = mean_norm * self._y_std_train + self._y_mean_train
            std_power = std_norm * self._y_std_train
        else:
            print("Note: Predictions are in normalized scale (zero-mean, unit-variance normalized during training)")
            print("      To denormalize, store y_mean and y_std running train_and_get_kernel()")
            mean_power = mean_norm
            std_power = std_norm
        
        if return_std:
            return mean_power, std_power
        else:
            return mean_power


    def plot_optimized_points(self):

        """
        Plot the optimized points for the bayesian quadrature on top of the joint probability
        """

        if not hasattr(self, 'optimized_points'):
            raise ValueError("Optimized points not found. Please run optimize_points_torch() first to obtain optimized points.")

        X_plot = self.optimized_points

        # Define wind speed and direction
        ws = np.linspace(3, 25, 360)
        wd = np.linspace(0, 2 * np.pi, 360, endpoint=False)

        # Create a meshgrid for all combinations
        WS, WD = np.meshgrid(ws, wd)

        # Compute joint probabilities for the grid points
        if self.kernel_type == "Periodic":
            X_pdf = np.column_stack((WS.ravel(), np.cos(WD.ravel()), np.sin(WD.ravel())))
            joint_pdf = np.array(self._X_probabilities(X_pdf)).reshape(WS.shape)
        else:
            X_pdf = np.column_stack((WS.ravel(), WD.ravel()))
            joint_pdf = np.array(self._X_probabilities(X_pdf)).reshape(WS.shape)
        joint_pdf = joint_pdf.reshape(WS.shape)  # Reshape to 360x360

        # Plot joint probability
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(np.rad2deg(WD), WS, joint_pdf, levels=20, cmap='viridis')
        plt.xlabel('Wind Direction (deg)')
        plt.ylabel('Wind Speed (m/s)')
        plt.title("Wind joint probabiilty")

        # Convert solution to m/s and degrees for plotting
        x_opt = X_plot[:,0]

        if self.kernel_type == "Periodic":
            y_opt = np.arctan2(X_plot[:,2], X_plot[:,1]) % (2*np.pi)
        else:
            y_opt = X_plot[:,1]

        # Plot optimized points on top of the joint probability
        plt.scatter(x=np.rad2deg(y_opt), y=x_opt, color="red", label="Optimized points")
        plt.legend()
        plt.show()


class WindFarmGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, n_dims=3):

        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=n_dims
            )
        )

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(
            mean_x,
            covar_x
        )
    

