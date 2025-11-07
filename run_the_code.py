import numpy as np
from helper_functions import *
import IP_models
from preprocessing import *

#  PARAMETERS  YOU CAN CHANGE THESE
NUM_LOCATIONS =  100    
BIKE_MAX = 50
COST_BIKE = 580
COST_STATION = 20_000
BUDGET = 400_000
DIST_MAX = 2_000


###### read in POIs 
pois = pd.read_csv("edinburgh_pois.csv")

poi_weights = pois['category'].apply(designate_weight)
coords = pois[['lat', 'lon']].to_numpy()

locations_gdf = cluster_and_get_centremost_points(coords, NUM_LOCATIONS, poi_weights)

hist_starts_gdf = get_historical_trips()

# Assign the historical trips to the nearest location
near_labs = assign_points_to_nearest_location(hist_starts_gdf, locations_gdf)

# Use the number of trips that started nearby as a proxy for "desire"/"demand"
locs_with_demand, counts = np.unique(near_labs, return_counts = True)

desire_for_locs = np.zeros(NUM_LOCATIONS)
desire_for_locs[locs_with_demand] = counts


# get the distance matrix from this precomputed one of 100x100
folder = path("good_example_data", "05 Nov 2025 03-50PM")
all_dists = np.load(path(folder, "all_dists.npy"))[:NUM_LOCATIONS,:NUM_LOCATIONS]

# Run the model
sol, mip_gap, near = IP_models.create_and_solve_model(
    desire_for_locs, all_dists,
    bike_max=BIKE_MAX, cost_bike=COST_BIKE, cost_station=COST_STATION, 
    budget=BUDGET, dist_max=DIST_MAX
)

print(sol[sol["desire"]>0])

# Colours: historical trips are the colour of the location they are assigned to
cmap = plt.colormaps['cool'].resampled( len(locs_with_demand) )
cols_for_locs_with_demand  = cmap(np.arange(len(locs_with_demand)))
cols_for_locations = np.zeros((NUM_LOCATIONS,4))
# locations with no demand/desire are shown in grey
cols_for_locations[:,:] = .5
cols_for_locations[:,3] = .5


cols_for_locations[locs_with_demand,:] = cols_for_locs_with_demand
cols_for_hist_data = cols_for_locations[near_labs]


# Plot the solution
built = sol["build"].to_numpy() == 1
_, ax1 = plt.subplots()

hist_starts_gdf.to_crs(locations_gdf.crs).plot(
    ax=ax1, label="Historical",
    color=cols_for_hist_data,
    markersize=10
    ,marker="."
)

locations_gdf[~built].plot(
    ax=ax1, label="Unused Locations",
    color=cols_for_locations[~built],
    markersize=30,
    marker = "x"
)

locations_gdf[built].plot(
    ax=ax1, label="Station Locations",
    color=cols_for_locations[built],
    markersize=40,
    marker = "^"
)

ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
ax1.legend()
plt.title("Colours show assignment")
plt.show()