#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 19:21:18 2025

@author: michael
"""
import pandas as pd 
import geopandas as gpd
import requests
import numpy as np
import math
import glob
from helper_functions import *
from sklearn import linear_model



def calculateCloseStations(stations, pois):

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
    stations_gdf['buffer_100m'] = stations_gdf.geometry.buffer(100)
    stations_gdf['buffer_250m'] = stations_gdf.geometry.buffer(250)
    stations_gdf['buffer_500m'] = stations_gdf.geometry.buffer(500)
    stations_gdf['buffer_700m'] = stations_gdf.geometry.buffer(700)

    #stations_gdf['buffer_1km'] = stations_gdf.geometry.buffer(1000)
    
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
    counts_100 = count_pois_within('buffer_100m')
    counts_250 = count_pois_within('buffer_250m')
    counts_500 = count_pois_within('buffer_500m')
    counts_700 = count_pois_within('buffer_700m')

    #counts_1km  = count_pois_within('buffer_1km')
    
    poi_counts = stations_gdf[['station_id']].set_index('station_id')
    poi_counts = poi_counts.join([counts_100, counts_250, counts_500, counts_700 ])
    poi_counts = poi_counts.reset_index()
    
    stations = pd.merge(stations, poi_counts, on = 'station_id')
    
    stations  = stations.dropna()
    return stations 


def Estimate_Demand(rep_points_df):
    """ takes df with lat and lon of the cluster centroid """
    
    stations = pd.read_csv( "station_data.csv")

    pois = pd.read_csv("edinburgh_pois.csv")

    filenames = glob.glob("*2021*.csv", root_dir="cyclehire-cleandata") #regex

    trips = pd.concat([ pd.read_csv(path("cyclehire-cleandata", f )) for f in filenames ])


    tripstartcount = trips['start_station_id'].value_counts().reset_index().rename(columns = {'count': 'Start_Trip_Counts'})
    endstartcount = trips['end_station_id'].value_counts().reset_index().rename(columns = {'count': 'end_Trip_Counts'})

    tripstartcount = tripstartcount.rename(columns={'start_station_id': 'station_id'})

    endstartcount = endstartcount.rename(columns={'end_station_id': 'station_id'})


    trip_count = pd.merge(tripstartcount, endstartcount, on = 'station_id')


    stations = pd.merge(stations, trip_count, on ='station_id')

    stations = calculateCloseStations(stations, pois)




    ## Get Clusters ####
    clustered_stations = rep_points_df 
    clustered_stations['station_id'] = np.arange(1, len(clustered_stations) +1 )
    clustered_stations = calculateCloseStations(clustered_stations, pois)
        
    exclude = ['station_id', 'name', 'address', 'rental_uris', 'lat', 'geometry', 'lon', 'capacity', 'Start_Trip_Counts', 'end_Trip_Counts']
    
    # X_cols = [
    #        'buffer_500m_commercial', 'buffer_500m_hospital', 'buffer_500m_library',
    #        'buffer_500m_residential', 'buffer_500m_school',
    #        'buffer_500m_university', 'buffer_1km_commercial',
    #        'buffer_1km_hospital', 'buffer_1km_library', 'buffer_1km_residential',
    #        'buffer_1km_school', 'buffer_1km_university']
    
    
    X_cols = [col for col in clustered_stations.columns if col not in exclude]
    
    y_cols = ['Start_Trip_Counts', 'end_Trip_Counts']
    
    
    X = stations[X_cols]
    
    #X = stations.drop(columns = exclude)

    
    
    predDemanddf = pd.DataFrame()

    predDemanddf['station_id'] =  np.arange(1, len(clustered_stations) +1 )

    for i, name in enumerate(y_cols):
        
        y = stations[y_cols[i]]
    
        model = linear_model.PoissonRegressor().fit(X, y)
        
        preddemand=  model.predict(clustered_stations[X_cols])


        predDemanddf[f'prediced_{name}'] = preddemand



    clustered_stations = pd.merge(clustered_stations, predDemanddf, on = 'station_id')
    return clustered_stations.drop(columns = X_cols)




def osrm_table_block(coord_block_src, coord_block_dst):
    """Query OSRM for one block pair: src-block → dst-block"""
    src_str = ";".join(f"{lon},{lat}" for lat, lon in coord_block_src)
    dst_str = ";".join(f"{lon},{lat}" for lat, lon in coord_block_dst)

    url = (
        f"http://router.project-osrm.org/table/v1/bike/{src_str}"
        f"?destinations={';'.join(str(i) for i in range(len(coord_block_dst)))}"
        f"&sources={';'.join(str(i) for i in range(len(coord_block_src)))}"
        f"&annotations=distance"
    )

    response = requests.get(url).json()
    return response["distances"]


def osrm_bike_matrix(coords):
    """
    coords: list of (lat, lon)
    returns: full NxN numpy distance matrix (N can be >100)
    """
    MAX_OSRM = 100
    n = len(coords)
    blocks = math.ceil(n / MAX_OSRM)

    # full matrix initialized
    full_matrix = np.zeros((n, n))

    for i in range(blocks):
        for j in range(blocks):
            # determine block slice
            src_start = i * MAX_OSRM
            src_end = min((i + 1) * MAX_OSRM, n)

            dst_start = j * MAX_OSRM
            dst_end = min((j + 1) * MAX_OSRM, n)

            src_block = coords[src_start:src_end]
            dst_block = coords[dst_start:dst_end]

            # query OSRM for this block
            block_matrix = osrm_table_block(src_block, dst_block)

            # insert into full matrix
            full_matrix[src_start:src_end, dst_start:dst_end] = block_matrix

    return full_matrix


# Haversine distance function
def haversine(u, v):
    R = 6371.0  # Earth radius in kilometers
    dlat = v[0] - u[0]
    dlon = v[1] - u[1]
    a = np.sin(dlat/2)**2 + np.cos(u[0]) * np.cos(v[0]) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

