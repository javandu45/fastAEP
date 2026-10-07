
# Obtain the wind rose for a location using the wind data from the netCDF file.
# The wind rose is saved to a CSV file, to be used later to obtain the PyWake
# Site object.

import xarray as xr
import pandas as pd
import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve

import warnings

warnings.filterwarnings("ignore")

def obtain_windrose(wind_farm):

    print(f"Obtaining wind rose for {wind_farm}...")

    data = xr.open_dataset(f'data/raw/{wind_farm}.nc')
    time = data.coords['time'].data
    data = data.to_dataframe().reset_index()
    wind_data = pd.DataFrame({"t": time, "ws": data["WS"], "wd": data["WD"]})

    # Create wind rose
    wind_data["ws"] = pd.to_numeric(wind_data["ws"], errors='coerce')
    wind_data["wd"] = pd.to_numeric(wind_data["wd"], errors='coerce')

    # Average wind speed
    print(f"Average wind speed for {wind_farm}: {wind_data['ws'].mean():.2f} m/s")

    sectors = {}
    nd = 360

    for i in range(1, nd+1):

        sectors[f"Sector_{i}"] = pd.DataFrame(columns=["t", "ws", "wd"])
        sectors["prob"] = []

        sectors[f"Sector_{i}"] = wind_data[(wind_data["wd"] > (i-1)*360/nd) & (wind_data["wd"] <= i*360/nd)]


    sectors["prob"] = [len(sectors[f"Sector_{i}"])/len(wind_data) for i in range(1, nd+1)]

    def gamma_func(k, mu_1, mu_2):

        return gamma(1+1/k)**2/gamma(1+2/k) - (mu_1**2)/mu_2

    def weibull_parameters(data):

        fst_moment = np.mean(data)
        snd_moment = np.mean(data**2)

        k = fsolve(gamma_func, 2, args=(fst_moment, snd_moment))

        A = fst_moment/gamma(1+1/k)

        return A, k

    sectors_sum = pd.DataFrame(columns=["A", "k", "weibull"])
    sectors_sum["A"] = nd*[0.]
    sectors_sum["k"] = nd*[0.]
    sectors_curve = {}

    U = np.linspace(0,30,100)
    weibull = np.zeros((nd,100))

    for i in range(1, nd+1):

        sectors_sum["A"].iloc[i-1], sectors_sum["k"].iloc[i-1] = weibull_parameters(sectors[f"Sector_{i}"]["ws"])
        
        sectors_sum["weibull"][i-1] = sectors_sum["k"][i-1] / U * (U / sectors_sum["A"][i-1])**sectors_sum["k"][i-1] * np.exp(-(U / sectors_sum["A"][i-1])**sectors_sum["k"][i-1])

    # Save wind rose in csv
    wind_char = pd.DataFrame({
        "P": sectors["prob"],
        "A": sectors_sum["A"],
        "k": sectors_sum["k"]
    })

    wind_char.to_csv(f"data/wind/{wind_farm}.csv", sep=",", index=False)

wind_farms_to_read = pd.read_csv("data/top_20_windfarms.csv")

for _, row in wind_farms_to_read.iterrows():
    wind_farm = row["name"]

    obtain_windrose(wind_farm)

