#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 20:07:45 2025

@author: michael
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 19:18:32 2025

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


######## Parameters 

num_clusters = 800

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



# get the old demand ####
hist_starts_gdf = get_historical_trips()
near_labs = assign_points_to_nearest_location(hist_starts_gdf, locations_gdf)

# Use the number of trips that started nearby as a proxy for "desire"/"demand"
locs_with_demand, counts = np.unique(near_labs, return_counts = True)

desire_for_locs = np.zeros(len(locations_gdf))
desire_for_locs[locs_with_demand] = counts

locations_gdf["old Demand"] = desire_for_locs
#####






#### Get Distances #########


lat_lon = np.radians(locations_gdf[['lat', 'lon']].to_numpy())  # Convert to radians

#dist = osrm_bike_matrix(locations_gdf[['lat', 'lon']].to_numpy())
dist = squareform(pdist(lat_lon, metric=haversine))


dist = dist * 1000

dist_mat = dist


###### get train stations ######

trainstations = pd.read_csv('trainstations.csv')

trainstations = np.radians(trainstations[['Latitude', 'Longitude']].to_numpy())  # Convert to radians

dist_to_trains = np.zeros((trainstations.shape[0], lat_lon.shape[0]))

for i in range(trainstations.shape[0]):
    for j in range(lat_lon.shape[0]):
        dist_to_trains[i, j] = haversine(trainstations[i], lat_lon[j])
 
dist_to_trains = dist_to_trains * 1000

#dist_to_trains = dist_to_trains[np.any(dist_to_trains < 200, axis=1)]





def create_and_solve_model(
    desire, dist_mat, bike_max, cost_bike, cost_station, budget, dist_to_trains,
    rev_per_bike, dist_min=100, dist_min_center=100, total_bikes_max=800,
    dist_max=5000, city_centre_radius=1000, dist_to_train_station=400
):
 
    xp.setOutputEnabled(False)
    prob = xp.problem("BikeStationOptimization")
    
    num_locs = desire.shape[0]
    I = range(num_locs)

    ########### Decision variables #############
    build = np.array([prob.addVariable(name=f"build_{i}", vartype=xp.binary) for i in I], dtype=xp.npvar)
    bikes = np.array([prob.addVariable(name=f"bikes_{i}", vartype=xp.integer) for i in I], dtype=xp.npvar)

    ########### Soft train-station coverage #############
    num_trains = dist_to_trains.shape[0]
    covered = np.array([prob.addVariable(name=f"covered_train_{t}", vartype=xp.binary) for t in range(num_trains)], dtype=xp.npvar)
    gamma = 10_000  # reward for covering a train station

    for t in range(num_trains):
        covered_indices = [i for i in I if dist_to_trains[t, i] <= dist_to_train_station]
        if len(covered_indices) > 0:
            prob.addConstraint(covered[t] <= xp.Sum(build[i] for i in covered_indices))
        else:
            prob.addConstraint(covered[t] == 0)

    ########### Objective function #############
    prob.setObjective(
        xp.Sum(desire[i] * bikes[i] for i in I) + gamma * xp.Sum(covered[t] for t in range(num_trains)),
        sense=xp.maximize
    )

    ########### Constraints #############

    # Max bikes per location and linking with build
    prob.addConstraint(bikes[i] <= bike_max * build[i] for i in I)
    prob.addConstraint(build[i] <= bikes[i] for i in I)
    prob.addConstraint(bikes[i] <= desire[i] + 1 for i in I)
    prob.addConstraint(xp.Sum(bikes[i] for i in I) <= total_bikes_max)

    # Budget constraint
    prob.addConstraint(xp.Sum(cost_bike*bikes[i] + cost_station*build[i] for i in I) <= budget)

    # City centre and distance constraints
    centre_lat_lon = np.radians(np.array([55.9486, -3.1999]))
    dist_to_centre = np.array([haversine(latlon, centre_lat_lon) for latlon in lat_lon]) * 1000
    is_city_center = dist_to_centre <= city_centre_radius
    near = dist_mat <= dist_max

    prob.addConstraint(build[i] <= xp.Sum(near[i,j] * build[j] for j in I if j != i) for i in I)

    too_close_center = np.zeros_like(dist_mat, dtype=int)
    too_close_outside = np.zeros_like(dist_mat, dtype=int)
    for i in I:
        for j in I:
            if i < j:
                if is_city_center[i] and is_city_center[j]:
                    if dist_mat[i,j] < dist_min_center:
                        too_close_center[i,j] = 1
                        too_close_center[j,i] = 1
                else:
                    if dist_mat[i,j] < dist_min:
                        too_close_outside[i,j] = 1
                        too_close_outside[j,i] = 1

    prob.addConstraint(build[i] + xp.Sum(too_close_center[i,j]*build[j] for j in I if j != i) <= 1 for i in I)
    prob.addConstraint(build[i] + xp.Sum(too_close_outside[i,j]*build[j] for j in I if j != i) <= 1 for i in I)

    ########### Solving ###########
    print("Solving")
    solve_start = perf_counter()
    prob.solve()
    solve_end = perf_counter()
    print(f"Solved in {solve_end-solve_start:.0f} seconds with {num_locs:,} variables")

    # MIP gap
    MIP_gap = get_MIP_gap(prob)
    print(f"{MIP_gap=:.2%}")

    # Extract solution
    solution = pd.DataFrame({
        "build": np.array([int(i) for i in prob.getSolution(build)]),
        "bikes": np.array([int(i) for i in prob.getSolution(bikes)]),
        "desire": desire,
        "city_center": is_city_center
    })

    return solution, MIP_gap



#demand = locations_gdf['prediced_Start_Trip_Counts'] + locations_gdf['prediced_end_Trip_Counts'] 
demand = locations_gdf['prediced_Start_Trip_Counts'] * 0.5 + locations_gdf['old Demand'] * 0.5

demand = demand/365


sol, mip = create_and_solve_model(
    desire=demand, dist_mat=dist, bike_max=30,dist_to_trains = dist_to_trains,
    cost_bike=580, cost_station=20_000, budget=2_000_000,rev_per_bike = 1000 ,dist_min_center = 250, 
    dist_min = 700, dist_max =1500, city_centre_radius = 1500)


finallocs = pd.concat([locations_gdf[['lat', 'lon']], sol], axis = 1)


df = finallocs

m = folium.Map(
    location=[df['lat'].mean(), df['lon'].mean()],
    zoom_start=13
)

# -----------------------------
# Add bike station markers
# -----------------------------
for i, row in df.iterrows():
    color = 'green' if row['build'] == 1 else 'red'
    max_bikes = df['bikes'].max()
    radius = 4 + (row['bikes'] / max_bikes) * 10

    popup_text = (
        f"<b>Build station:</b> {bool(row['build'])}<br>"
        f"<b>Bikes:</b> {row['bikes']}<br>"
        f"<b>Demand:</b> {row['desire']}<br>"
        f"<b>City Centre:</b> {'Yes' if row['city_center'] else 'No'}<br>"
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

# -----------------------------
# Add train station markers (NumPy array)
# -----------------------------
# trainstations = np.array([[lat, lon], [lat, lon], ...])
trainstations_deg = np.degrees(trainstations)

for lat, lon in trainstations_deg:
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color='blue', icon='train', prefix='fa'),
        popup=f"<b>Train Station</b><br>Lat: {lat}<br>Lon: {lon}"
    ).add_to(m)

# -----------------------------

m.save('bike_stations.html')

# Optional: add heatmap for demand
heat_data = df[['lat', 'lon', 'desire']].values.tolist()
HeatMap(heat_data, radius=20, blur=15, max_zoom=13).add_to(m)


