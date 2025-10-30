#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 20:23:22 2025

@author: michael
"""

from sklearn.cluster import KMeans
from shapely.geometry import MultiPoint, Point
from geopy.distance import great_circle
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

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

