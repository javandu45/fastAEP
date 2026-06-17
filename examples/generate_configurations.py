import itertools, json
from pathlib import Path
import pandas as pd

# Read wind farm data
base_dir = Path(__file__).resolve().parent.parent
wf = pd.read_csv(base_dir / "data" / "top_20_windfarms.csv")

farms = list(wf["name"])
farms.remove("Kriegers_Flak_K2-K3") # Remove for now, since it is two different boundaries

aep_methods = {
    "360_WD": 8, "72_WD": 8, "Average_WS": 8,
    "Uniform_CT": 8, "FLOWERS": 1, "BQ": 8,
    "RQ": 8,                            
}

aep_methods = {"FLOWERS": 1}

wake_models = ["NOJ", "Gaussian", "TurbOPark"]

tasks = []
for farm, (method, n_cpu), wake in itertools.product(
        farms, aep_methods.items(), wake_models):
    tasks.append({
        "farm_id": farm,
        "aep_method": method,
        "wake_model": wake,
        "n_cpu": n_cpu,
        "start_id": 0,
    })

with open("configurations.json", "w") as f:
    json.dump(tasks, f, indent=2)