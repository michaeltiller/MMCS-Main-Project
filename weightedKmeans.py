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
import contextily as ctx
from time import perf_counter
import requests, ujson


########### get the potential locations for the bike stations ##############
# We are going to base our locations off of the POIs
# There are too many (30,000) to consider all of them
# We will cluster them using weighted K-means
# Then in each cluster, we take one point that is closest to the cluster's centroid and discard the rest
# These are our potential locations for the stations

# Coordinates
pois = pd.read_csv("edinburgh_pois.csv")

def designate_weight(cat):
    """designate a POI's weight based off it's category"""
    if cat in ("university", "commercial"):
        return 3
    elif cat in ("school", "hospital"):
        return 2
    elif cat in ("residential",  "library"):
        return 1
    else:
        return 1

# Sorry Michael i just hated this code
# pois['weights'] = 1
# pois['weights'] = np.where(pois['category'] == 'library', 1, pois['weights'])
# pois['weights'] = np.where(pois['category'] == 'residential', 1, pois['weights'])
# pois['weights'] = np.where(pois['category'] == 'hospital', 2, pois['weights'])
# pois['weights'] = np.where(pois['category'] == 'school', 2, pois['weights'])


# pois['weights'] = np.where(pois['category'] == 'university', 3, pois['weights'])
# pois['weights'] = np.where(pois['category'] == 'commercial', 3, pois['weights'])
#The output of the new way is the same
# assert ( pois['weights'] == pois['category'].apply(designate_weight) ).all()

poi_weights = pois['category'].apply(designate_weight)

#go from df to np ndarray
coords = pois[['lat', 'lon']].to_numpy()



num_clusters = 30
s = perf_counter()
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords, sample_weight=poi_weights)
e = perf_counter()
print(f"KMeans took {e-s:.0f} seconds")
cluster_labels = kmeans.labels_



clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

s = perf_counter()
centermost_points = clusters.map(get_centermost_point)
e = perf_counter()
print(f"getting centremost points took {e-s:.0f} seconds")

rep_points_df = pd.DataFrame({
    "lat":centermost_points.map(lambda tup : tup[0]), 
    "lon":centermost_points.map(lambda tup : tup[1])
    })

# # Convert to GeoDataFrame to plot a picture 
rep_points_gdf = gpd.GeoDataFrame(
    rep_points_df, 
    geometry=[Point(xy) for xy in zip(rep_points_df['lon'], rep_points_df['lat'])],
    crs="EPSG:4326"  # WGS84 Latitude/Longitude
).to_crs(epsg=3857)

ax = rep_points_gdf.plot(
    figsize=(8, 8),
    color='red',
    markersize=20,    # smaller dots
    alpha=0.2,        # more transparency
)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.title("Points Of Interest in Edinburgh")
plt.show()

########### find which locations are "close" to each other #####################
# we will use the osrm module for this 
# it has a web API which takes a url with the longatudes and latitudes 
# and returns a distance table
def dist_from_point_basic(p, arr):
    """get the distance from a single point ``p`` to each point in ``arr```
       ``arr`` must be small enough to pass into the url string for osrm
    """

    url = "http://router.project-osrm.org/table/v1/bike/"
    #add p as the first coordinates
    url += f"{p[1]},{p[0]};"
    #its longitude first  <------------------
    url += ";".join( f"{lon},{lat}" for lat, lon in arr )
    url += "?sources=0&annotations=distance"

    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception("fuck " + resp.text) 

    # stored as a list within a list because there is only one source point
    dists = ujson.loads(resp.content)["distances"][0]
    # An annoyance is that in each chunk the first value is the dist from p to p
    # so trim that out 
    # this is only because p is added into the front of the array every time so it keeps calculating that
    return np.array(dists[1:])


def get_distances(coords_arr):
    n_rows = coords_arr.shape[0]
    dists = np.zeros((n_rows,n_rows))

    #the we can only do so many co-ordinates at once -> break it up into batches
    MAX_BATCH_SIZE = 325 #works, 330 is too large, try rise it further if you want
    num_batches = np.ceil(n_rows/MAX_BATCH_SIZE)
    # split the array into a list of num_batches arrays
    # each small array has at most MAX_BATCH_SIZE elements
    batched = np.array_split(coords_arr, num_batches)

    
    # i was trying to doing something more effecient than this using threading
    # but i got blocked, I believe for sending more than one request per minute
    print("On:")
    for i in range(n_rows):
        print(i)
        dists[i,] = np.concat( [ dist_from_point_basic(coords_arr[i], batch) for batch in batched ] )
    print()
    return dists


# print("Starting calculating distances")
# s = perf_counter()
# all_dists = get_distances(centermost_points.to_numpy())
# e = perf_counter()
# print(f"It took {e-s:.0f} seconds with {centermost_points.shape[0]:,} records")
# # all_dists = x = np.load("dists 60 590 secs.npy")
# ACCEPTABLE_DISTANCE = 5_000 # metres
# near = all_dists <= ACCEPTABLE_DISTANCE


# ########### get the demand for each potential location ##############
# # read the trip data from csv into dataframes
#for now only look at 2021 data
filenames = glob.glob("*2021*.csv", root_dir="cyclehire-cleandata") #regex
trips = pd.concat([ pd.read_csv(path("cyclehire-cleandata", f )) for f in filenames  ])

# # we consider a start location and an end location as two different "units" of demand
# # i.e. there is a demand of one wherever the trip started and a demand of one wherever the trip ended
col_names = ["long","lati"]
starts = trips[['start_station_longitude', 'start_station_latitude' ]]
starts.columns = col_names

ends = trips[['end_station_longitude','end_station_latitude' ]]
ends.columns = col_names
demand = pd.concat([starts,ends], ignore_index=True)

# assign the demand to the nearest station
s = perf_counter()

dist_tree = cKDTree(rep_points_df)
_, nearest_loc = dist_tree.query(demand, k = 1)

e = perf_counter()
print(f"Assigning historical demand to locations took {e-s:.0f} seconds")

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

stamp = timestamp()
print(f"Writing to file with timestamp:{stamp}")

os.mkdir(path(DATA_FOLDER, stamp))
np.save(path(DATA_FOLDER, stamp, "demand"), demand_per_loc)
# np.save(path(DATA_FOLDER, stamp, "near"), near)

