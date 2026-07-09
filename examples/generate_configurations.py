import itertools, json
from pathlib import Path
import pandas as pd

# Read wind farm data
base_dir = Path(__file__).resolve().parent.parent
wf = pd.read_csv(base_dir / "data" / "top_20_windfarms.csv")

farms = list(wf["name"])
farms.remove("Kriegers_Flak_K2-K3")  # Remove for now, since it is two different boundaries
farms.remove("Gwynt_y_Mor")

# Wind farms to study
farms = ["Hornsea_Project_3_HOW03", "Thor","Hornsea_Project_2_-_Phase_1_Breesea", "Sofia"]
# farms = ["East_Anglia_TWO", "Thor", "Sofia"]
# farms = ["Hornsea_Project_3_HOW03"]


# AEP methods to study, with number of CPUs to use for each method
aep_methods = {
    "FLOWERS": 1, "72_WD": 8, "Average_WS": 8,
    "Uniform_CT": 8, "BQ": 8, "360_WD": 8,
    "RQ": 8, "SGD": 4
}
aep_methods = {"FLOWERS": 1}

# Wake models to study
wake_models = ["TurbOPark"]

# Number of multi starts considered
n_multistarts = 20

# Tolerance for each (aep_method, wake_model) combination
tolerances = {
    ("360_WD",     "NOJ"): 1e-6,
    ("72_WD",      "NOJ"): 1e-7,
    ("Average_WS", "NOJ"): 1e-6,
    ("Uniform_CT", "NOJ"): 1e-6,
    ("FLOWERS",    "NOJ"): 1e-6,
    ("BQ",         "NOJ"): 1e-7,
    ("RQ",         "NOJ"): 1e-7,
    ("SGD",        "NOJ"): 1,
    ("360_WD",     "Gaussian"): 1e-5,
    ("72_WD",      "Gaussian"): 1e-5,
    ("Average_WS", "Gaussian"): 1e-5,
    ("Uniform_CT", "Gaussian"): 1e-5,
    ("FLOWERS",    "Gaussian"): 1e-6,
    ("BQ",         "Gaussian"): 1e-6,
    ("RQ",         "Gaussian"): 1e-6,
    ("SGD",        "Gaussian"): 1,
    ("360_WD",     "TurbOPark"): 1e-6,
    ("72_WD",      "TurbOPark"): 1e-7,
    ("Average_WS", "TurbOPark"): 1e-6,
    ("Uniform_CT", "TurbOPark"): 1e-6,
    ("FLOWERS",    "TurbOPark"): 1e-6,
    ("BQ",         "TurbOPark"): 1e-7,
    ("RQ",         "TurbOPark"): 1e-7,
    ("SGD",        "TurbOPark"): 1,
}

tasks = []
for farm, (method, n_cpu), wake in itertools.product(
        farms, aep_methods.items(), wake_models):

    key = (method, wake)
    if key not in tolerances:
        raise KeyError(f"No tolerance defined for combination {key}")

    for i in range(n_multistarts):

        tasks.append({
            "farm_id":    farm,
            "aep_method": method,
            "wake_model": wake,
            "n_cpu":      n_cpu,
            "tol":        tolerances[key],
            "start_id":   i,
        })

    # UNCOMMENT FOR SEVERAL TOLERANCES
    # for tol in tolerances[key]:
    #     tasks.append({
    #         "farm_id":    farm,
    #         "aep_method": method,
    #         "wake_model": wake,
    #         "n_cpu":      n_cpu,
    #         "tol":        tol,
    #         "start_id":   0,
    #     })

print(f"Generated {len(tasks)} tasks")

with open(f"configurations_{wake_models[0]}.json", "w") as f:
    json.dump(tasks, f, indent=2)