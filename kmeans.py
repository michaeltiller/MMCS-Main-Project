# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 20:35:33 2025

@author: micha
"""
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

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
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.title("Points in Edinburgh")
plt.show()








coords = pois[['lat', 'lon']].to_numpy()





kms_per_radian = 6371.0088
epsilon = 0.25 / kms_per_radian
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
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.title("Points in Edinburgh")
plt.show()
