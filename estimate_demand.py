#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 13:12:35 2025

@author: michael
"""
import glob, os 
import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
import seaborn as sns
import folium
from folium.plugins import HeatMap


directory = "/Users/michael/Desktop/MMCS Project/"

stations = pd.read_csv(directory + "station_data.csv")

pois = pd.read_csv(directory + "edinburgh_pois.csv")

filenames = glob.glob("*2021*.csv", root_dir=directory +"cyclehire-cleandata") #regex

trips = pd.concat([ pd.read_csv(path(directory +"cyclehire-cleandata", f )) for f in filenames  ])


tripstartcount = trips['start_station_id'].value_counts().reset_index().rename(columns = {'count': 'Start_Trip_Counts'})
endstartcount = trips['end_station_id'].value_counts().reset_index().rename(columns = {'count': 'end_Trip_Counts'})

tripstartcount = tripstartcount.rename(columns={'start_station_id': 'station_id'})

endstartcount = endstartcount.rename(columns={'end_station_id': 'station_id'})


trip_count = pd.merge(tripstartcount, endstartcount, on = 'station_id')


stations = pd.merge(stations, trip_count, on ='station_id')


####


def calculateCloseStations(stations):

    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.lon, stations.lat),
        crs='EPSG:4326'  
    )
    
    pois_gdf = gpd.GeoDataFrame(
        pois,
        geometry=gpd.points_from_xy(pois.lon, pois.lat),
        crs='EPSG:4326'
    )
    
    stations_gdf = stations_gdf.to_crs(epsg=3857)
    pois_gdf = pois_gdf.to_crs(epsg=3857)
    
    stations_gdf['buffer_500m'] = stations_gdf.geometry.buffer(500)
    stations_gdf['buffer_1km'] = stations_gdf.geometry.buffer(1000)
    
    def count_pois_within(buffer_col):
        station_buffer_gdf = stations_gdf[['station_id', buffer_col]].copy()
        station_buffer_gdf = station_buffer_gdf.set_geometry(buffer_col)
        station_buffer_gdf.crs = pois_gdf.crs  
        joined = gpd.sjoin(pois_gdf, station_buffer_gdf, predicate='within')
    
        return (
            joined.groupby(['station_id', 'category'])
            .size()
            .unstack(fill_value=0)
            .add_prefix(f'{buffer_col}_')
        )
    
    counts_500 = count_pois_within('buffer_500m')
    counts_1km  = count_pois_within('buffer_1km')
    
    poi_counts = stations_gdf[['station_id']].set_index('station_id')
    poi_counts = poi_counts.join([counts_500, counts_1km])
    poi_counts = poi_counts.reset_index()
    
    stations = pd.merge(stations, poi_counts, on = 'station_id')
    
    stations  = stations.dropna()
    return stations 

stations= calculateCloseStations(stations)

X_cols = [
       'buffer_500m_commercial', 'buffer_500m_hospital', 'buffer_500m_library',
       'buffer_500m_residential', 'buffer_500m_school',
       'buffer_500m_university', 'buffer_1km_commercial',
       'buffer_1km_hospital', 'buffer_1km_library', 'buffer_1km_residential',
       'buffer_1km_school', 'buffer_1km_university']

y_cols = ['Start_Trip_Counts', 'end_Trip_Counts']


X = stations[X_cols]
y = stations[y_cols[1]]


#model = LinearRegression().fit(X, y)
model = linear_model.PoissonRegressor().fit(X, y)
model.coef_
model.intercept_


#model = GradientBoostingRegressor().fit(X,y)

######### Predict Cluster ##########################

clustered_stations = rep_points_df 
clustered_stations['station_id'] = np.arange(1, len(clustered_stations) +1 )
clustered_stations = calculateCloseStations(clustered_stations)


preddemand=  model.predict(clustered_stations[X_cols])

#predDemand = pd.DataFrame(preddemand, columns = y_cols)
predDemanddf = pd.DataFrame()

predDemanddf['station_id'] =  np.arange(1, len(clustered_stations) +1 )
predDemanddf['Start_Trip_Counts'] = preddemand



clustered_stations = pd.merge(clustered_stations, predDemanddf, on = 'station_id')

prob_grid = clustered_stations.pivot_table(
    index='lat',
    columns='lon',
    values='Start_Trip_Counts'
)


sns.heatmap(
    prob_grid.sort_index(ascending=False),
    cmap='coolwarm',
    cbar_kws={'label': 'Predicted Probability'},
    linewidths=0,       #remove lines between cells
    linecolor=None,     
)
plt.title('demand Heatmap over Space')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xticks([], [])
plt.yticks([], [])
plt.show()



m = folium.Map(location=[55.9533, -3.1883], zoom_start=13)

lat_vals = prob_grid.index.values
lon_vals = prob_grid.columns.values
heat_vals = prob_grid.values

data = [
    [lat_vals[i], lon_vals[j], heat_vals[i, j]]
    for i in range(len(lat_vals))
    for j in range(len(lon_vals))
    if not np.isnan(heat_vals[i, j])
]

HeatMap(data).add_to(m)

m.save("edinburgh_heatmap.html")

