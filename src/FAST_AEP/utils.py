import pandas as pd
import numpy as np
import random
from shapely.geometry import Point, Polygon
from scipy.special import gamma
import json
import re

from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site import UniformWeibullSite



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

    limits = pd.read_csv(f"data/boundaries/{wind_farm}.csv", index_col=0)
    limits = np.array(limits)

    return limits


def get_wind_farm_data(wind_farm):

    wfs = pd.read_csv(f"data/top_20_windfarms.csv", index_col=0)
    wfs.reset_index(inplace=True)
    
    wf_data = wfs[wfs['name'] == wind_farm]

    return wf_data