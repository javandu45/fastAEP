from FAST_AEP.basic_wfm import WD_Bins, average_WS, uniform_CT
from FAST_AEP.FLOWERS import NOJ_flowers, gaussian_flowers, TurbOPark_flowers
from FAST_AEP.BQ import bayesian_quadrature
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.deficit_models import TurboNOJDeficit
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014

import time
import pandas as pd
from utils import generic_site
import warnings
warnings.filterwarnings("ignore")

# Define wind farm
site = generic_site(10)
turbines = V80()
x, y = Hornsrev1Site().initial_position.T

def setup_BQ_wfm(deficit_model, aep_method):

    if deficit_model == "NOJ":
        wfm = Jensen_1983(site, turbines)
    elif deficit_model == "Gaussian":
        wfm = Bastankhah_PorteAgel_2014(site, turbines, k=0.05)
    elif deficit_model == "Turbopark":
        deficit_model = TurboNOJDeficit()
        wfm = PropagateDownwind(site, turbines,
                                wake_deficitModel=deficit_model,
                                superpositionModel=SquaredSum(),
                                rotorAvgModel=AreaOverlapAvgModel())

    BQ_wfm = bayesian_quadrature(site=site,
                                 windTurbines=turbines,
                                 flow_model=wfm,
                                 x0=x,
                                 y0=y,
                                 N_train=3000,
                                 N_MC = 4000,
                                 aep_method=aep_method)
    
    BQ_wfm.train_and_get_kernel()
    BQ_wfm.optimize_BQ_points(N_points=360, N_attempts=1)

    return BQ_wfm

# Setting up BQ models
# NOJ
BQ_NOJ = setup_BQ_wfm("NOJ", "BQ")
RQ_NOJ = setup_BQ_wfm("NOJ", "RQ")

# Gaussian
BQ_Gaussian = setup_BQ_wfm("Gaussian", "BQ")
RQ_Gaussian = setup_BQ_wfm("Gaussian", "RQ")

# TurbOPark
BQ_TurbOPark = setup_BQ_wfm("Turbopark", "BQ")
RQ_TurbOPark = setup_BQ_wfm("Turbopark", "RQ")

###################
# NOJ
###################

aep_models = ["360 WD", "72 WD", "Average WS", "Uniform CT", "FLOWERS", "BQ", "RQ"]
times_NOJ = []

for i, aep_model in enumerate(aep_models):

    if aep_model == "360 WD":
        wfm = WD_Bins(site, turbines, "NOJ", n_bins=360)
    elif aep_model == "72 WD":
        wfm = WD_Bins(site, turbines, "NOJ", n_bins=72)
    elif aep_model == "Average WS":
        wfm = average_WS(site, turbines, "NOJ")
    elif aep_model == "Uniform CT":
        wfm = uniform_CT(site, turbines, "NOJ")
    elif aep_model == "FLOWERS":
        wfm = NOJ_flowers(site, turbines, n_terms=10)
    elif aep_model == "BQ":
        wfm = BQ_NOJ
    elif aep_model == "RQ":
        wfm = RQ_NOJ

    start_time = time.time()
    AEP = wfm.aep(x, y)
    end_time = time.time()

    times_NOJ.append(end_time - start_time)

###################
# Gaussian
###################

times_Gaussian = []

for i, aep_model in enumerate(aep_models):

    if aep_model == "360 WD":
        wfm = WD_Bins(site, turbines, "Gaussian", n_bins=360)
    elif aep_model == "72 WD":
        wfm = WD_Bins(site, turbines, "Gaussian", n_bins=72)
    elif aep_model == "Average WS":
        wfm = average_WS(site, turbines, "Gaussian")
    elif aep_model == "Uniform CT":
        wfm = uniform_CT(site, turbines, "Gaussian")
    elif aep_model == "FLOWERS":
        wfm = gaussian_flowers(site, turbines, n_terms=10)
    elif aep_model == "BQ":
        wfm = BQ_Gaussian
    elif aep_model == "RQ":
        wfm = RQ_Gaussian

    start_time = time.time()
    AEP = wfm.aep(x, y)
    end_time = time.time()

    times_Gaussian.append(end_time - start_time)

###################
# TurbOPark
###################

times_TurbOPark = []

for i, aep_model in enumerate(aep_models):

    if aep_model == "360 WD":
        wfm = WD_Bins(site, turbines, "Turbopark", n_bins=360)
    elif aep_model == "72 WD":
        wfm = WD_Bins(site, turbines, "Turbopark", n_bins=72)
    elif aep_model == "Average WS":
        wfm = average_WS(site, turbines, "Turbopark")
    elif aep_model == "Uniform CT":
        wfm = uniform_CT(site, turbines, "Turbopark")
    elif aep_model == "FLOWERS":
        wfm = TurbOPark_flowers(site, turbines, n_terms=10)
    elif aep_model == "BQ":
        wfm = BQ_TurbOPark
    elif aep_model == "RQ":
        wfm = RQ_TurbOPark

    start_time = time.time()
    AEP = wfm.aep(x, y)
    end_time = time.time()

    times_TurbOPark.append(end_time - start_time)


# Print table of results
results_df = pd.DataFrame({
    "AEP Model": aep_models,
    "NOJ Time (s)": times_NOJ,
    "Gaussian Time (s)": times_Gaussian,
    "TurbOPark Time (s)": times_TurbOPark
})
print(results_df)
