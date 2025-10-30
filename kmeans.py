# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 20:35:33 2025

@author: micha
"""
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import geopandas as gpd
#import contextily as 
import os
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from scipy.spatial import cKDTree


pois = pd.read_csv("edinburgh_pois.csv")

df = pois

gdf  = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)


ax = gdf.plot(
    figsize=(8, 8),
    color='red',
    markersize=20,    # smaller dots
    alpha=0.2,        # more transparency
)
#ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.title("Points in Edinburgh")
plt.show()



coords = pois[['lat', 'lon']].to_numpy()


kms_per_radian = 6371.0088
epsilon = 0.1 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)
centermost_points = clusters.map(get_centermost_point)

lats, lons = zip(*centermost_points)
rep_points = pd.DataFrame({'lon':lons, 'lat':lats})


kmeandf  = gpd.GeoDataFrame(rep_points, geometry=gpd.points_from_xy(rep_points.lon, rep_points.lat), crs="EPSG:4326")
kmeandf = kmeandf.to_crs(epsg=3857)

ax = kmeandf.plot(
    figsize=(8, 8),
    color='red',
    markersize=20,    # smaller dots
    alpha=0.2,        # more transparency
)
#ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.title("Points in Edinburgh")
plt.show()


######## Get Demand #####################

# Create new station ID's
kmeandf['station_id'] = kmeandf.index + 1

# Load the Count Files
count_files = os.listdir('cyclehire-cleandata')

files = pd.DataFrame({'file': count_files})

files[['year', 'month']] = files['file'].str.extract(r'(\d{4})_(\d{2})')
files['year'] = files['year'].astype(int)
files['month'] = files['month'].astype(int)
files = files.sort_values(['year', 'month']).reset_index(drop=True)

files = files.loc[files['year'] == 2021]

CycleHireData = pd.DataFrame()

for filename in files['file']:
    temp = pd.read_csv(f'cyclehire-cleandata/{filename}')
    CycleHireData = pd.concat([CycleHireData, temp])



starting_stations = CycleHireData[['start_station_latitude','start_station_longitude' ]]
starting_stations.columns = ['station_lat', 'station_lon']

ending_stations = CycleHireData[['end_station_latitude','end_station_longitude' ]]
ending_stations.columns = ['station_lat', 'station_lon']


totaltrips = pd.concat([starting_stations, ending_stations ])


# get closest cluster 
tree = cKDTree(kmeandf[['lat', 'lon']])
distances, indices = tree.query(totaltrips, k=1)

totaltrips['nearest_station_id'] = kmeandf.iloc[indices]['station_id'].values

tripCounts = pd.DataFrame(totaltrips[['nearest_station_id']].value_counts()).reset_index()


kmeandf = pd.merge(kmeandf, tripCounts, left_on = 'station_id', right_on = 'nearest_station_id', how = 'left') 





