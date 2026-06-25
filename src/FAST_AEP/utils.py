import pandas as pd
import numpy as np
import random
from shapely.geometry import Point, Polygon
from scipy.special import gamma
import json
import re
from pathlib import Path
import h5py

from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site import UniformWeibullSite
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.deficit_models import TurboNOJDeficit
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.utils.gradients import autograd

import FAST_AEP.basic_wfm as basic_wfm
import FAST_AEP.FLOWERS as FLOWERS
from FAST_AEP.BQ import bayesian_quadrature


def generic_site(wind_farm):

    """
    Generate a UniformWeibullSite based on wind speed characteristics from a CSV file.
    """

    wind_char = pd.read_csv(f"data/wind/{wind_farm}.csv", index_col=0)

    wind_char = wind_char.reset_index()

    site = UniformWeibullSite(
        p_wd = wind_char["P"],
        a = wind_char["A"],
        k = wind_char["k"],
        ti = 0.1    
    )

    return site


def generate_array(n_tur, turbine, spacing=5, limits="Square"):

    """
    Generate a grid array of wind turbines based on the number of turbines, turbine diameter, and spacing factor.
    """
    
    if limits == "Square":
        D = turbine.diameter()
        n_gaps = int(np.ceil(np.sqrt(n_tur)))  # Ensure enough gaps for 100 turbines
        x = np.arange(0, n_gaps * D * spacing, D * spacing)
        y = np.arange(0, n_gaps * D * spacing, D * spacing)
        x, y = np.meshgrid(x, y)
        x = x.flatten()[:n_tur] 
        y = y.flatten()[:n_tur]

        return x.flatten(), y.flatten()
    

def generate_random_array(n_tur, turbine, spacing=5, limits="Square", seed=None):

    """
    Generate a random array of wind turbines based on the number of turbines, turbine diameter, and spacing factor.
    """

    if isinstance(limits, str) and limits == "Square":
        D = turbine.diameter()
        n_gaps = int(np.ceil(np.sqrt(n_tur))) 
        x_limits = (0, n_gaps * D * spacing)
        y_limits = (0, n_gaps * D * spacing)

        if seed is not None:
            random.seed(int(seed))

        # Generate random x and y positions
        x = [random.uniform(x_limits[0], x_limits[1]) for _ in range(n_tur)]
        y = [random.uniform(y_limits[0], y_limits[1]) for _ in range(n_tur)]

        return np.array(x), np.array(y)
    
    # Calculate the number of points (turbines)
    else: 

        n_points = n_tur

        # Create the polygon
        polygon = Polygon(limits)

        # Get the bounds of the polygon
        min_x, min_y, max_x, max_y = polygon.bounds

        # Generate random points within the bounding box with minimum distance
        random_points = []
        min_distance = turbine.diameter() * spacing  # Minimum distance between turbines

        # Set the random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        while len(random_points) < n_points:
            random_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
            if polygon.contains(random_point):
            # Check if the point is far enough from existing points
                if all(random_point.distance(Point(p.x, p.y)) >= min_distance for p in random_points):
                    random_points.append(random_point)

        # Convert the list of points to a numpy array
        init_coords = np.array([[point.x, point.y] for point in random_points])

        return init_coords[:, 0], init_coords[:, 1]
    

def average_ws_site(ws=11):

    """
    Get the average wind speed and probability distribution from a CSV file.
    """

    wind_char = pd.read_csv(f"wind_roses/wind_rose_{ws}.csv", index_col=0)

    wind_char = wind_char.reset_index()

    avg_ws = wind_char['A'] * gamma(1 + 1/wind_char['k'])
    P = wind_char['P']

    return P, avg_ws


def turbine_generator(wind_farm):

    """
    Generate pywake wind turbine object for the selected wind farm based on json file containing
    turbine power, hub height, and rotor diameter.
    """

    with open(f"data/turbines/{wind_farm}.json", 'r') as f:
        turbine_data = json.load(f)

    turbine = GenericWindTurbine(
        name=turbine_data["turbine_model"],
        diameter=turbine_data["rotor_diameter_display"],
        hub_height=turbine_data["hub_height_display"],
        power_norm=turbine_data["rated_power_display"]*1000
    )

    return turbine


def get_limits(wind_farm):
    """Get wind farm limits from CSV file based on wind farm name"""

    limits = pd.read_csv(f"data/boundaries/{wind_farm}.csv", index_col=0)
    limits = np.array(limits)

    return limits


def get_wind_farm_data(wind_farm):
    """Get wind farm data from CSV file based on wind farm name"""

    wfs = pd.read_csv(f"data/top_20_windfarms.csv", index_col=0)
    wfs.reset_index(inplace=True)
    
    wf_data = wfs[wfs['name'] == wind_farm]

    return wf_data


def build_wfm(site, windTurbines, deficit_model = "NOJ", k = 0.05):
    """Builds a pywake wind farm model based on the selected deficit model"""

    if deficit_model == "NOJ":
        wfm = Jensen_1983(site, windTurbines, k=k)
    elif deficit_model == "Gaussian":
        wfm = Bastankhah_PorteAgel_2014(site, windTurbines, k=k)
    elif deficit_model == "TurbOPark":
        deficit_model = TurboNOJDeficit()
        wfm = PropagateDownwind(site, windTurbines,
                                wake_deficitModel=deficit_model,
                                superpositionModel=SquaredSum(),
                                rotorAvgModel=AreaOverlapAvgModel())

    return wfm


def build_aep_model(aep_method, wind_farm, deficit_model, k = 0.05, n_cpu=1):
    """Builds an AEP model based on the selected AEP method and deficit model"""

    site = generic_site(wind_farm)
    windTurbines = turbine_generator(wind_farm)
    wfm = build_wfm(site, windTurbines, deficit_model, k)
    wf_data = get_wind_farm_data(wind_farm)
    n_turbines = wf_data['turbine_count'].values[0]
    limits = get_limits(wind_farm)

    if aep_method == "360_WD":
        aep_model = basic_wfm.WD_Bins(site=site, windTurbines=windTurbines, deficit_model=deficit_model, n_bins=360, k=k)
    elif aep_method == "72_WD":
        aep_model = basic_wfm.WD_Bins(site=site, windTurbines=windTurbines, deficit_model=deficit_model, n_bins=72, k=k)
    elif aep_method == "Average_WS":
        aep_model = basic_wfm.average_WS(site=site, windTurbines=windTurbines, deficit_model=deficit_model, k=k)
    elif aep_method == "Uniform_CT":
        aep_model = basic_wfm.uniform_CT(site=site, windTurbines=windTurbines, deficit_model=deficit_model, k=k)
    elif aep_method == "FLOWERS":
        if deficit_model == "NOJ":
            aep_model = FLOWERS.NOJ_flowers(site=site, windTurbines=windTurbines, n_terms=10, k=k)
        elif deficit_model == "Gaussian":
            aep_model = FLOWERS.gaussian_flowers(site=site, windTurbines=windTurbines, n_terms=10, k=k)
        elif deficit_model == "TurbOPark":
            aep_model = FLOWERS.TurbOPark_flowers(site=site, windTurbines=windTurbines, n_terms=10)
    elif aep_method == "BQ":
        x, y = generate_random_array(n_tur=n_turbines, turbine=windTurbines, spacing=2, limits=limits, seed=55)
        aep_model = bayesian_quadrature(site=site,
                                       windTurbines=windTurbines,
                                       flow_model=wfm,
                                       x0=x,
                                       y0=y,
                                       N_train=3000,
                                       N_MC = 4000,
                                       aep_method=aep_method)
        aep_model.train_and_get_kernel()
        aep_model.optimize_BQ_points(N_points=360, N_attempts=10, jitter=0.1)
        aep_model.setup_gradients(gradient_method=autograd, n_cpu=n_cpu)
    elif aep_method == "RQ":
        x, y = generate_random_array(n_tur=n_turbines, turbine=windTurbines, spacing=2, limits=limits, seed=55)
        aep_model = bayesian_quadrature(site=site,
                                       windTurbines=windTurbines,
                                       flow_model=wfm,
                                       x0=x,
                                       y0=y,
                                       N_train=3000,
                                       N_MC = 4000,
                                       aep_method=aep_method)
        aep_model.train_and_get_kernel()
        aep_model.optimize_BQ_points(N_points=360, N_attempts=10, jitter=0.1)
        aep_model.setup_gradients(gradient_method=autograd, n_cpu=n_cpu)
    return aep_model


def save_results_in_H5(farm_id, aep_method, wake_model, start_id, res):

    out_path = Path("results") / "optimization" / f"windfarm_{farm_id}.h5"
    out_path.parent.mkdir(exist_ok=True)

    # The key is just a path-like string inside the HDF5 file
    group_key = f"{aep_method}/{wake_model}/start_{start_id}"

    print(f"Saving results to {out_path} under group {group_key}...")

    with h5py.File(out_path, "a") as f:

        if group_key in f:
            del f[group_key]

        grp = f.create_group(group_key)

        # Datasets: arrays and scalars
        grp.create_dataset("x_opt", data = res["x_opt"]) 
        grp.create_dataset("y_opt", data=res["y_opt"])
        grp.create_dataset("convergence", data=res["convergence"])

        # Attributes: small scalars and strings (not arrays)
        grp.attrs["aep_final"] = res["aep_final"]
        grp.attrs["time"] = res["time"]
        grp.attrs["n_iter"] = res["n_iter"]
        grp.attrs["success"] = res["success"]
