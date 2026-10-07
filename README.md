# Fast AEP

This repository contains the code used to obtain the results for the paper "A comparison of efficient AEP models for Wind Farm Layout Optimization". It includes the different AEP models used (PyWake based, FLOWERS, Bayesian Quadrature (BQ) and Optimal-design Weighted Quadrature (OWQ)), and the WFLO algorithm Stochastic Gradient Descent (SGD). 

This repository was developed by Javier Andueza Bueno from DTU Wind and Energy Systems.

## The FAST_AEP package

The repository's main component is the simple FAST_AEP package (`src/FAST_AEP`) or `pip install -e .`, which is composed of:

- `basic_wfm.py`: containing the more conventional wind farm models, derived from PyWake. These are the Baseline and Coarse WD (`WD_bins`), Average WS (`average_WS`) and Uniform $C_T$ (`Uniform_CT`).
- `FLOWERS.py`: where the code for the different FLOWERS models can be found, for NO Jensen (`NOJ_flowers`), Bastankhah Porte-Agel (`gaussian_flowers`) and Nygaard TurbOPark (`TurbOPark_flowers`).
- `BQ.py`: containing a single class for the BQ for AEP implementation, which can also be used for OWQ.
- `SGD.py`: with a quick set up for the AEP and gradients needed in the SGD algorithm.

Each module includes a function to obtain AEP and its gradients. Besides these modules for the AEP models, two extra modules are included:

- `optimization.py`: containing the optimization problem set up, regardless of the wind farm model used, for a specific wind farm and wake model.
- `utils.py`: this file includes several auxilliary functions needed to read files, build the wind farm models and the optimization problems.

## Running the optimization

The `generate_configurations.py` file generates the json file to be used by the `optimize_config.py`, which runs a single configuration of AEP model, wind farm and wake model, and stores in in the H5 file.

All other functions have secondary purposes. Please refer to the specific file to see what their function is.

## Data

In the data folder, the different wind farm data, including wind roses, boundaries and the respective wind turbine can be found. These correspond to the data regarding the 4 used wind farms. The wind time series are obtained from the [New European Wind Atlas](https://map.neweuropeanwindatlas.eu/), while the boundaries and other data are downloaded from [Boundary Layer](https://www.boundary-layer.com/).