from FAST_AEP.basic_wfm import BasicWFM

from py_wake.utils.gradients import autograd
import numpy as np

class SGD(BasicWFM):

    """
    Class to set up an SGD "wind farm model"
    """

    def __init__(self, site, windTurbines, deficit_model, k=0.05, n_samples=50):

        super().__init__(site, windTurbines, deficit_model, k=k)

        self.name = "SGD"
        self.n_samples = n_samples

        # Extract wind data from the site for sampling
        self.wd = site.default_wd
        self.freqs = site.ds.Sector_frequency.values[:-1]
        self.A = site.ds.Weibull_A.values[:-1]
        self.K = site.ds.Weibull_k.values[:-1]


    def _sampling(self):

        scenario = np.random.choice(self.wd, size=self.n_samples, p=self.freqs)
        A_scenario = self.A[scenario.astype(int)]
        K_scenario = self.K[scenario.astype(int)]

        wd_scenario = scenario
        ws_scenario = np.random.weibull(K_scenario) * A_scenario

        return wd_scenario, ws_scenario
    

    def aep(self, x, y, n_cpu=1):

        AEP = 0

        return float(AEP)


    def aep_gradient(self, x, y, gradient_method=autograd, n_cpu=1):

        wd, ws = self._sampling()  # _sampling returns (wd, ws)

        jx, jy = self.wfm.aep_gradients(gradient_method=gradient_method,
                                               x=x,
                                               y=y,
                                               ws=ws,
                                               wd=wd,
                                               time=True,
                                               n_cpu=n_cpu)

        daep = np.array([np.atleast_2d(jx), np.atleast_2d(jy)])

        return daep
    