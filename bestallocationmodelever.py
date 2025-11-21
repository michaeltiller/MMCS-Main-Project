#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:18:12 2025

@author: michael
"""


import numpy as np
from helper_functions import *
import IP_models
from preprocessing import *
from demandfuncs import *
import pandas as pd 
import glob
import xpress as xp
import folium
from folium import CircleMarker
from folium.plugins import HeatMap
from scipy.spatial.distance import pdist, squareform
import itertools

######## Parameters 

num_clusters = 400

######## 

pois = pd.read_csv("edinburgh_pois.csv")
poi_weights = pois['category'].apply(designate_weight)
coords = pois[['lat', 'lon']].to_numpy()


#### Calculate Clusters ###
rep_point_df = cluster_and_get_centremost_points(coords, num_clusters, kmeans_weights=poi_weights)



#### add estimate demand ####

rep_point_df = Estimate_Demand(rep_point_df)


locations_gdf = gpd.GeoDataFrame(
    rep_point_df,
    geometry=gpd.points_from_xy(rep_point_df.lon, rep_point_df.lat),
    crs='EPSG:4326'  
).to_crs(epsg=3857)

loc_lon, loc_lat = locations_gdf["lon"].to_numpy(), locations_gdf["lat"].to_numpy()



####### get the old demand #######
hist_starts_gdf = get_historical_trips()
near_labs = assign_points_to_nearest_location(hist_starts_gdf, locations_gdf)

# Use the number of trips that started nearby as a proxy for "desire"/"demand"
locs_with_demand, counts = np.unique(near_labs, return_counts = True)

desire_for_locs = np.zeros(len(locations_gdf))
desire_for_locs[locs_with_demand] = counts

locations_gdf["old Demand"] = desire_for_locs

#### Get Distances #########
dist_mat = get_dists_gps(locations_gdf)

###### get train stations ######

train_stations = pd.read_csv('trainstations.csv')
num_trains = train_stations.shape[0]

near_to_trains = np.zeros((num_trains, num_clusters), dtype = bool)

train_lon, train_lat = train_stations["Longitude"].to_numpy(), train_stations["Latitude"].to_numpy()
for t in range(num_trains):
    near_to_trains[t] = haversine_np(
        train_lon[t], train_lat[t],
        loc_lon, loc_lat
        )
    near_to_trains = near_to_trains < .4 #400 metres
    print(f"Locs near train station {t} is {near_to_trains[t].sum()}")





demand = locations_gdf['prediced_Start_Trip_Counts'] * 0.25 + locations_gdf['old Demand'] * 0.25 + predict_bike_count_MLP(locations_gdf[["lat", "lon"]].to_numpy())* .5


demand = demand/365


# sol, mip, alloc_df, arcs = IP_models.create_and_solve_extended_model(
#     desire=demand, dist_mat=dist_mat, bike_max=35,
#     cost_bike=580, cost_station=20_000, budget=2_000_000,rev_per_bike = 1000 ,
#     near_to_trains=near_to_trains,
#     dist_min = 0.4, dist_max =2)
sol, mip, alloc_df, arcs = IP_models.create_and_solve_extended_model(
    desire=demand, dist_mat=dist_mat, bike_max=30,
    cost_bike=1000, cost_station=5000, budget=2_000_000, rev_per_bike = 1000,
    near_to_trains=near_to_trains,
    dist_min = 0.4, dist_max =2)

df = pd.concat([locations_gdf[['lat', 'lon']], sol], axis = 1)

m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=13)

# Add station markers
for i, row in df.iterrows():
    color = 'green' if row['build'] == 1 else 'red'
    max_bikes = df['bikes'].max()
    radius = 4 + (row['bikes'] / max_bikes) * 10  # scales between 4–14 px
    popup_text = (
        f"<b>Build station:</b> {bool(row['build'])}<br>"
        f"<b>Bikes:</b> {row['bikes']}<br>"
        f"<b>Demand:</b> {row['desire']}<br>"
        f"<b>Index:</b> {i}"
    )

    CircleMarker(
        location=[row['lat'], row['lon']],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=folium.Popup(popup_text, max_width=250)
    ).add_to(m)

alloc_mat = alloc_df.values

for i in range(len(df)):
    for j in range(len(df)):
        if alloc_mat[i, j] == 1:
            # Coordinates of the two stations
            lat_i, lon_i = df.loc[i, ['lat', 'lon']]
            lat_j, lon_j = df.loc[j, ['lat', 'lon']]

            # Add a polyline for the allocation
            folium.PolyLine(
                locations=[(lat_i, lon_i), (lat_j, lon_j)],
                color="blue",
                weight=3,
                opacity=0.6,
                tooltip=f"{i} → {j}"
            ).add_to(m)
            
            
            


# for lat, lon in zip( np.degrees(train_lat), np.degrees(train_lon) ):
#     folium.Marker(
#         location=[lat, lon],
#         icon=folium.Icon(color='blue', icon='train', prefix='fa'),
#         popup=f"<b>Train Station</b><br>Lat: {lat}<br>Lon: {lon}"
#     ).add_to(m)

for lat, lon in zip( train_lat, train_lon ):
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color='blue', icon='train', prefix='fa'),
        popup=f"<b>Train Station</b><br>Lat: {lat}<br>Lon: {lon}"
    ).add_to(m)
# Optional: add heatmap for demand
# heat_data = df[['lat', 'lon', 'desire']].values.tolist()
# HeatMap(heat_data, radius=20, blur=15, max_zoom=13).add_to(m)

m.show_in_browser()

stamp = timestamp()
save_folder = path("good_example_data", stamp)

os.mkdir(save_folder)
os.chdir(save_folder)

m.save("the_map.html")
sol.to_csv("the_sol.csv")
pd.DataFrame({
    "demand":demand
}).to_csv("demand.csv")





