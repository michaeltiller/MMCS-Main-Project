# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 19:49:10 2025

@author: micha
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



#### Get Distances #########


lat_lon = np.radians(locations_gdf[['lat', 'lon']].to_numpy())  # Convert to radians

#dist = osrm_bike_matrix(locations_gdf[['lat', 'lon']].to_numpy())
dist = squareform(pdist(lat_lon, metric=haversine))
dist = dist * 1000

dist_mat = dist

def create_and_solve_model(desire, dist_mat, bike_max, cost_bike, cost_station, budget, rev_per_bike,dist_min =100, dist_max=5_000):

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
    
    # allocated = [prob.addVariable(name=f'allocated_{i}_{j}', vartype=xp.binary)
    #          for i in I for j in I]
    
    allocated = np.array(
    [[prob.addVariable(name=f'allocated_{i}_{j}', vartype=xp.binary) for j in I]
     for i in I]
    )
    
    u = np.array(
    [prob.addVariable(name = f"u_{i}", vartype = xp.integer, lb=0, ub=num_locs) for i in I]
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
    
    
    for i in I:
        for j in I:
            prob.addConstraint(allocated[i,j] <= build[i])
            prob.addConstraint(allocated[i,j] <= build[j])
    
    prob.addConstraint(
        build[j] == xp.Sum(allocated[j, i] for i in I) for j in I 
    )
    
    prob.addConstraint(
        allocated[i, j] + allocated[j, i] <= 1 for i in I for j in I
    )
    
    # Distance constraints
    prob.addConstraint( 
        build[j] * dist_min <= xp.Sum(allocated[j, i] * dist_mat[i, j] for i in I)
        for j in I 
    )
    prob.addConstraint( 
        build[j] * dist_max >= xp.Sum(allocated[j, i] * dist_mat[i, j] for i in I)
        for j in I 
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
        "desire": desire
    })
    
    
    alloc_solution = np.zeros_like(allocated, dtype=int)

    temp = prob.getSolution(allocated)
    
    for i in I:
        for j in I:
            alloc_solution[i, j] = int(temp[i, j])

    # Put into a DataFrame for readability
    alloc_df = pd.DataFrame(
        alloc_solution,
        columns=[f"to_{j}" for j in I],
        index=[f"from_{i}" for i in I]
    )
        
   
    # return the pertient info that was not inputted
    return solution, MIP_gap, alloc_df


demand = locations_gdf['prediced_Start_Trip_Counts'] + locations_gdf['prediced_end_Trip_Counts'] 

sol, mip, alloc_df = create_and_solve_model(
    desire=demand, dist_mat=dist, bike_max=50,
    cost_bike=580, cost_station=20_000, budget=2_000_000,rev_per_bike = 1000 ,
    dist_min = 700, dist_max =6000)


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
m.save('bike_stations.html')


bad_edges = []
for i in range(len(df)):
    for j in range(len(df)):
        if alloc_mat[i,j] == 1 and (df.loc[i,'build']==0 or df.loc[j,'build']==0):
            bad_edges.append((i,j))

bad_edges

# Optional: add heatmap for demand
heat_data = df[['lat', 'lon', 'desire']].values.tolist()
HeatMap(heat_data, radius=20, blur=15, max_zoom=13).add_to(m)


