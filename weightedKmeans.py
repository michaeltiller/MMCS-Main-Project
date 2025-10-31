from sklearn.cluster import KMeans
from shapely.geometry import MultiPoint, Point
from geopy.distance import great_circle
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from helper_functions import *
import  glob, os
from scipy.spatial import cKDTree


########### get the potential locations for the bike stations ##############
# We are going to base our locations off of the POIs
# There are too many (30,000) to consider all of them
# We will cluster them using weighted K-means
# Then in each cluster, we take one point that is closest to the cluster's centroid and discard the rest
# These are our potential locations for the stations

# Coordinates
pois = pd.read_csv("edinburgh_pois.csv")
pois['weights'] = 1
pois['weights'] = np.where(pois['category'] == 'library', 1, pois['weights'])
pois['weights'] = np.where(pois['category'] == 'residential', 1, pois['weights'])
pois['weights'] = np.where(pois['category'] == 'hospital', 2, pois['weights'])
pois['weights'] = np.where(pois['category'] == 'school', 2, pois['weights'])


pois['weights'] = np.where(pois['category'] == 'university', 3, pois['weights'])
pois['weights'] = np.where(pois['category'] == 'commercial', 3, pois['weights'])

coords = pois[['lat', 'lon']].to_numpy()



num_clusters = 60  

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords, sample_weight=pois['weights'].to_numpy())
cluster_labels = kmeans.labels_



clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

centermost_points = clusters.map(get_centermost_point)

lats, lons = zip(*centermost_points)
rep_points_df = pd.DataFrame({'lon': lons, 'lat': lats})

# Convert to GeoDataFrame
rep_points_gdf = gpd.GeoDataFrame(
    rep_points_df, 
    geometry=[Point(xy) for xy in zip(rep_points_df['lon'], rep_points_df['lat'])],
    crs="EPSG:4326"  # WGS84 Latitude/Longitude
)

ax = rep_points_gdf.plot(
    figsize=(8, 8),
    color='red',
    markersize=20,    # smaller dots
    alpha=0.2,        # more transparency
)
#ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.title("Points Of Interest in Edinburgh")
plt.show()



########### get the demand for each potential location ##############
# read the trip data from csv into dataframes
#for now only look at 2021 data
filenames = glob.glob("*2021*.csv", root_dir="cyclehire-cleandata") #regex
trips = pd.concat([ pd.read_csv(path("cyclehire-cleandata", f ),nrows =1) for f in filenames  ])

# we consider a start location and an end location as two different "units" of demand
# i.e. there is a demand of one wherever the trip started and a demand of one wherever the trip ended
col_names = ["long","lati"]
starts = trips[['start_station_longitude', 'start_station_latitude' ]]
starts.columns = col_names

ends = trips[['end_station_longitude','end_station_latitude' ]]
ends.columns = col_names
demand = pd.concat([starts,ends], ignore_index=True)

# assign the demand to the nearest station
dist_tree = cKDTree(rep_points_df)
_, nearest_loc = dist_tree.query(demand, k = 1)

# sum the demand per station
locs_with_demand = np.unique_counts(nearest_loc)
demand_per_loc = np.zeros(shape=rep_points_df.shape[0], dtype = int)

demand_per_loc[locs_with_demand.values] = locs_with_demand.counts


########## write it to a file so we can use it in an OR problem ###########
# if we are doing this a lot i want to store the data we used each time
# # i want to store it in its own subfolder in a data folder
DATA_FOLDER = "gen_data"
if not os.path.exists(DATA_FOLDER): 
    os.mkdir(DATA_FOLDER)
folder = path(DATA_FOLDER, timestamp())
os.mkdir(folder)

np.save(path(folder, "demand"), demand_per_loc)

