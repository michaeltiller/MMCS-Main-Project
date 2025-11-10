import numpy as np
from helper_functions import *
import IP_models
from preprocessing import *


pois = pd.read_csv("edinburgh_pois.csv")
poi_weights = pois['category'].apply(designate_weight)
coords = pois[['lat', 'lon']].to_numpy()

show_elbow_of_weighted_kmeans(coords, range(1, 50+1), poi_weights)
num_regions = 10

locs2_gdf = gpd.GeoDataFrame({
    "region": KMeans(n_clusters=num_regions, random_state=0).fit(coords, sample_weight=poi_weights).labels_ ,
    "lat": coords[:,0],
    "lon": coords[:,1]
    },
    geometry=[Point(lon, lat) for lat, lon in coords],
    crs="EPSG:4326"  # WGS84 Latitude/Longitude
).to_crs(epsg=3857)

# Assign the historical trips to the nearest location
hist_starts_gdf = get_historical_trips()
near_labs = assign_points_to_nearest_location(hist_starts_gdf, locs2_gdf)

# Use the number of trips that started nearby as a proxy for "desire"/"demand"
locs_with_demand, counts = np.unique(near_labs, return_counts = True)

desire_for_locs = np.zeros(len(coords))
desire_for_locs[locs_with_demand] = counts

locs2_gdf["demand"] = desire_for_locs
# I'm guessing the demand gets mapped to the nearest POI to each station
print(f"Number of locations with demand: { np.sum(locs2_gdf["demand"]>0) :,}  out of {locs2_gdf.shape[0]:,}" )

regions_gdf = locs2_gdf.dissolve(by="region", aggfunc={"demand":"sum"})
print(regions_gdf["demand"])

col_regions, col_region_points = create_colours_for_locs_and_assignments(
    np.arange(num_regions), locs2_gdf["region"]
)

_, ax = plt.subplots()

regions_gdf.convex_hull.plot(ax=ax, alpha = .2, color=col_regions)
regions_gdf.convex_hull.boundary.plot(ax=ax, color="black")
regions_gdf.plot(ax=ax, color = col_region_points)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.show()


