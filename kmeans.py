# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 20:35:33 2025

@author: micha
"""

pois = pd.read_csv("edinburgh_pois.csv")


coords = df.as_matrix(columns=['lat', 'lon'])
db = DBSCAN(eps=eps, min_samples=ms, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))