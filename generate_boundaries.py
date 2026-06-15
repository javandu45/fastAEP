import geopandas as gpd

all_wf = gpd.read_file("data/raw/boundary-layer-active-wind-farms.gpkg")

# Filter to those that have been comissioned
all_wf = all_wf[
	all_wf["commissioning_date"].notnull()
	& (all_wf["commissioning_date"] != "Not confirmed")
]

# Filter those with the turbine count
all_wf = all_wf[
    all_wf["turbine_count"].notnull()
    & (all_wf["turbine_count"] != "Not confirmed")
]

# Filter those with the turbine model
all_wf = all_wf[
    all_wf["turbine_model"].notnull()
    & (all_wf["turbine_model"] != "Not confirmed")
]

# Sort them in descending order of size
all_wf = all_wf.sort_values(by="capacity_mw", ascending=False)

# Select the top 20
top_20_wf = all_wf.head(20)

# Pick only name, country, centroid_lng, centroid_lat, capacity_mw, turbine_count, turbine_model,
# rated_power_display, rotor_diameter_display, hub_height_display and geometry

interesting_data = ["name", "country", "centroid_lng", "centroid_lat", "capacity_mw", "turbine_count", "turbine_model",
                    "rated_power_display", "rotor_diameter_display", "hub_height_display", "geometry"]

top_20_wf = top_20_wf[interesting_data]

# Apply the same name cleanup to the wind farms dataset
top_20_wf["name"] = top_20_wf["name"].astype(str).str.replace(" ", "_", regex=False)
top_20_wf["name"] = top_20_wf["name"].str.replace(r"[\(\)\[\]\{\}]", "", regex=True)
top_20_wf["name"] = top_20_wf["name"].str.replace(r"[^A-Za-z0-9_-]+", "_", regex=True).str.strip("_")

# Save the boundary coordinates of each wind farm to its own CSV file
import os
import re
import pandas as pd

output_dir = "data/boundaries"
os.makedirs(output_dir, exist_ok=True)

# Clean rated_power_display: remove trailing 'MW' and convert to float where possible
if "rated_power_display" in top_20_wf.columns:
    top_20_wf["rated_power_display"] = (
        top_20_wf["rated_power_display"].astype(str)
        .str.replace("MW", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", pd.NA)
        .astype(float)
    )

# Clean rotor_diameter_display: remove trailing 'm' and convert to float where possible
if "rotor_diameter_display" in top_20_wf.columns:
    top_20_wf["rotor_diameter_display"] = (
        top_20_wf["rotor_diameter_display"].astype(str)
        .str.replace("m", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", pd.NA)
        .astype(float)
    )

# Clean hub_height_display: set "Not confirmed" values to 150 and convert to float
if "hub_height_display" in top_20_wf.columns:
    def extract_hub_height(val):
        if pd.isna(val) or val == "Not confirmed":
            return 150.0
        val_str = str(val)
        # Extract first number (integer or decimal)
        match = re.search(r"\d+(?:\.\d+)?", val_str)
        return float(match.group()) if match else 150.0
    
    top_20_wf["hub_height_display"] = top_20_wf["hub_height_display"].apply(extract_hub_height)





# Save boundaries in CSV file
for _, row in top_20_wf.iterrows():
    # Estimate the best UTM zone for this specific geometry
    geom_series = gpd.GeoSeries([row["geometry"]], crs=top_20_wf.crs)
    utm_crs = geom_series.estimate_utm_crs()
    geom_utm = geom_series.to_crs(utm_crs).iloc[0]

    rows = []
    for poly_idx, polygon in enumerate(geom_utm.geoms):
        x, y = polygon.exterior.coords.xy
        for x_i, y_i in zip(x, y):
            rows.append({"polygon": poly_idx, "x": x_i, "y": y_i})

    out_path = os.path.join(output_dir, f"{row['name']}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)


# Plot boundaries of the top 20 wind farms
import matplotlib.pyplot as plt

# Plot the boundaries of the top 20 wind farms
fig, ax = plt.subplots(5, 4, figsize=(12, 10))
ax = ax.ravel()

for i, (_, row) in enumerate(top_20_wf.iterrows()):
    if i >= len(ax):
        break
    ax_i = ax[i]
    for polygon in row["geometry"].geoms:
        x, y = polygon.exterior.coords.xy
        ax_i.plot(x, y, label=row["name"])
    ax_i.set_title(row["name"])
    ax_i.set_aspect("equal", adjustable="box")
    ax_i.set_xlabel("")
    ax_i.set_ylabel("")
    ax_i.set_xticks([])
    ax_i.set_yticks([])

for ax_i in ax[len(top_20_wf):]:
    ax_i.axis("off")

plt.tight_layout()

top_20_wf_df = pd.DataFrame(top_20_wf)
top_20_wf_df = top_20_wf_df.drop(columns=["geometry"])


# Save turbines for each wind farm in json file
import json

output_dir = "data/turbines"
os.makedirs(output_dir, exist_ok=True)

for _, row in top_20_wf.iterrows():
    turbine_data = {
        "turbine_model": row["turbine_model"],
        "rated_power_display": row["rated_power_display"],
        "rotor_diameter_display": row["rotor_diameter_display"],
        "hub_height_display": row["hub_height_display"]
    }
    
    json_path = os.path.join(output_dir, f"{row['name']}.json")
    
    with open(json_path, "w") as f:
        json.dump(turbine_data, f, indent=2)

