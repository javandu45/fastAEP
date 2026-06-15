from abc import ABC, abstractmethod

from py_wake.literature.noj import Jensen_1983
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.deficit_models import NOJDeficit, BastankhahGaussianDeficit, TurboNOJDeficit
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.superposition_models import SquaredSum, LinearSum
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.wind_farm_models.engineering_models import All2All
from py_wake.utils.gradients import autograd

import numpy as np
from scipy.special import gamma

class BasicWFM(ABC):

    """
    Base class for PyWake based simplified wind farm models.
    """

    def __init__(self, site, windTurbines, deficit_model, k=0.05, n_cpu=1):

        self.n_cpu = n_cpu
        self.site = site

        if deficit_model == "NOJ":
            self.wfm = Jensen_1983(site, windTurbines, k=k)
        elif deficit_model == "Gaussian":
            self.wfm = Bastankhah_PorteAgel_2014(site, windTurbines, k=k)
        elif deficit_model == "Turbopark":
            deficit_model = TurboNOJDeficit()
            self.wfm = PropagateDownwind(site, windTurbines,
                                         wake_deficitModel=deficit_model,
                                         superpositionModel=SquaredSum(),
                                         rotorAvgModel=AreaOverlapAvgModel())

        else:
            raise ValueError(f"Deficit model {deficit_model} not supported. Choose from 'NOJ', 'Gaussian', or 'Turbopark'.")
        
    @abstractmethod
    def aep(self, x, y):
        """
        Calculate AEP for the given wind farm
        """
    
    @abstractmethod
    def aep_gradient(self, gradient_method=autograd):
        """
        Calculate AEP gradient for the given wind farm
        """


class WD_Bins(BasicWFM):

    """
    Wind farm model using a determined number of wind direction bins.
    """

    def __init__(self, site, windTurbines, deficit_model, k=0.05, n_cpu=1, n_bins=360):

        super().__init__(site, windTurbines, deficit_model, k=k, n_cpu=n_cpu)

        self.n_bins = n_bins


    def aep(self, x, y):

        wd = np.linspace(0,360,self.n_bins, endpoint=False)

        sim_res = self.wfm(x, y,
                            wd=wd,
                            ws=self.site.default_ws,
                            n_cpu=self.n_cpu)

        AEP = sim_res.aep().sum()

        return float(AEP)


    def aep_gradient(self, x, y, gradient_method=autograd):

        wd = np.linspace(0,360,self.n_bins, endpoint=False)

        jx, jy = self.wfm.aep_gradients(gradient_method=gradient_method,
                                               x=x, 
                                               y=y,
                                               ws=self.site.default_ws, 
                                               wd=wd,
                                               n_cpu=self.n_cpu)
                                               
        daep = np.array([np.atleast_2d(jx), np.atleast_2d(jy)])

        return daep
    

class average_WS(BasicWFM):

    """
    Wind farm model using an average wind speed for each wind direction bin
    """

    def __init__(self, site, windTurbines, deficit_model, k=0.05, n_cpu=1):

        super().__init__(site, windTurbines, deficit_model, k=k, n_cpu=n_cpu)

        self.avg_ws = site.ds.Weibull_A.values[:-1] * gamma(1 + 1/site.ds.Weibull_k.values[:-1])
        self.freqs = site.ds.Sector_frequency.values[:-1]
        self.freqs = self.freqs / sum(self.freqs)


    def aep(self, x, y):

        time_stamp = np.arange(360)

        wind_directions = np.linspace(0,360,360, endpoint=False)

        sim_res_avg_ws = self.wfm(x, y,
                                wd=wind_directions,
                                ws=self.avg_ws,
                                time=time_stamp,
                                n_cpu=self.n_cpu)

        power_vector = np.sum(np.array(sim_res_avg_ws.Power), axis=0)
        AEP = np.sum(power_vector * self.freqs) * 1e-9 * 8760

        return float(AEP)
    
    
    def aep_gradient(self, x, y, gradient_method=autograd):

        wd = np.linspace(0,360,self.n_bins, endpoint=False)

        jx, jy = self.wfm.aep_gradients(gradient_method=gradient_method,
                                        wrt_arg=['x', 'y'],
                                        x=x, 
                                        y=y,
                                        ws=self.avg_ws, 
                                        wd=wd,
                                        time=True,
                                        n_cpu=self.n_cpu)
                                               
        daep = np.array([np.atleast_2d(jx), np.atleast_2d(jy)])

        return daep
    

class uniform_CT(average_WS):

    """
    Wind farm model using a uniform CT for each wind direction bin, as well as average wind speed.

    Same as average_WS, but changing flow model
    """

    def __init__(self, site, windTurbines, deficit_model, k=0.05, n_cpu=1):

        super().__init__(site, windTurbines, deficit_model, k=k, n_cpu=n_cpu)

        if deficit_model == "NOJ":
            deficit_model = NOJDeficit(k=k)
            self.wfm = All2All(site, windTurbines,
                                wake_deficitModel=deficit_model,
                                superpositionModel=SquaredSum(),
                                rotorAvgModel=AreaOverlapAvgModel())
            
        elif deficit_model == "Gaussian":
            deficit_model = BastankhahGaussianDeficit(k=k)
            self.wfm = All2All(site, windTurbines,
                                wake_deficitModel=deficit_model,
                                superpositionModel=LinearSum())
            
        elif deficit_model == "Turbopark":
            deficit_model = TurboNOJDeficit()
            self.wfm = All2All(site, windTurbines,
                                wake_deficitModel=deficit_model,
                                superpositionModel=SquaredSum(),
                                rotorAvgModel=AreaOverlapAvgModel())
            

