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

def create_and_solve_model(desire, dist_mat, bike_max, cost_bike, cost_station, budget, rev_per_bike,dist_min =100, dist_min_center = 100, dist_max=5_000, city_centre_radius= 1000):

    # stop the big stream of text
    xp.setOutputEnabled(False)
    prob = xp.problem("First_bike_extension") 

    num_locs = desire.shape[0]
    I = range(num_locs)

    ########### Decision variables #############

    # we have the binary decision variable to build a station at location i
    build = np.array(
        [prob.addVariable(name = f"build_{i}", vartype = xp.binary) for i in I ]
        ,dtype = xp.npvar
    )
    #we have the integer decision variable of how many bikes to place at location i
    bikes = np.array(
        [prob.addVariable(name = f"bikes_{i}", vartype= xp.integer) for i in I]
        , dtype=xp.npvar
    )
    

    
   
    ########### objective function #############
 
    prob.setObjective(
        xp.Sum( (rev_per_bike * desire[i] - cost_bike) * bikes[i] for i in I ) - xp.Sum(cost_station * build[i] for i in I ) 
        , sense = xp.maximize
    )
    ########### constraints #############

    # we can place at most bikes_max bikes in each location
    # and if we put bikes somewhere then we must build a station there
    prob.addConstraint(
        bikes[i] <= bike_max*build[i] for i in I
    )

    # If we build a station somewhere then we must put at least one bike there
    prob.addConstraint(
        build[i] <= bikes[i] for i in I
    )
    
    # we will not put more bikes in a location than there is desire for bikes
    # the plus one is to avoid issues with combining the above constraint and the connectedness constraint
    prob.addConstraint(
        bikes[i] <= desire[i] + 1 for i in I
    )

        
    # stay within budget
    prob.addConstraint(
        xp.Sum( cost_bike*bikes[i] + cost_station*build[i] for i in I) <= budget
    )
    
    
    centre_lat_lon = np.radians(np.array([55.9486, -3.1999]))
    dist_to_centre = np.array([haversine(latlon, centre_lat_lon) for latlon in lat_lon]) * 1000 
    
    is_city_center = dist_to_centre <= city_centre_radius
    
    near = dist_mat <= dist_max
    
    
    prob.addConstraint(
        build[i] <= xp.Sum( near[i,j] *build[j] for j in I if j != i)
        for i in I
    )
    
    # Seperate min distance constraint for inside city centre 
    
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

    # City centre min-distance constraint
    prob.addConstraint(
        build[i] + xp.Sum(too_close_center[i,j]*build[j] for j in I if j != i) <= 1
        for i in I
    )
    # Outside / mixed min-distance constraint
    prob.addConstraint(
        build[i] + xp.Sum(too_close_outside[i,j]*build[j] for j in I if j != i) <= 1
        for i in I
    )

    
 
    ########## Solving ###########
    # Write problem statement to file, for debugging
    # prob.write("problem","ips")
 
    print("Solving")
    solve_start = perf_counter()
    prob.solve()
    solve_end = perf_counter()
    print(f"Solved in {solve_end-solve_start:.0f} seconds with {desire.shape[0]:,} variables")

    #mip gap 
    MIP_gap= get_MIP_gap(prob)
    print(f"{MIP_gap=:.2%}")

    # look at the solution
    solution  = pd.DataFrame({
        "build": np.array([ int(i) for i in prob.getSolution(build) ]),
        "bikes": np.array([int(i) for i in prob.getSolution(bikes) ]),
        "desire": desire,
        "city_center": is_city_center

    })
    
    

    # return the pertient info that was not inputted
    return solution, MIP_gap


#demand = locations_gdf['prediced_Start_Trip_Counts'] + locations_gdf['prediced_end_Trip_Counts'] 
demand = locations_gdf['prediced_Start_Trip_Counts'] * 0.5 + locations_gdf['old Demand'] * 0.5


sol, mip = create_and_solve_model(
    desire=demand, dist_mat=dist, bike_max=50,
    cost_bike=580, cost_station=20_000, budget=2_000_000,rev_per_bike = 1000 ,dist_min_center = 500, 
    dist_min = 1000, dist_max =1500, city_centre_radius = 2000)


finallocs = pd.concat([locations_gdf[['lat', 'lon']], sol], axis = 1)


df = finallocs

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


m.save('bike_stations.html')


# Optional: add heatmap for demand
heat_data = df[['lat', 'lon', 'desire']].values.tolist()
HeatMap(heat_data, radius=20, blur=15, max_zoom=13).add_to(m)


