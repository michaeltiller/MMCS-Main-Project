import requests
import ujson
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import time
import matplotlib.pyplot as plt
from helper_functions import *
import contextily as ctx
from time import perf_counter

# https://roadtraffic.dft.gov.uk/docs/index.html


def send_request(url, params=None, return_json=True):
    """
    A wrapper to send GET requests to ``URL`` with optional ``params``.
    This will return a ``Response`` object unless ``return_json=True`` in which case
    a conversion is attempted.
    """
    #slow request rate
    time.sleep(1)
    # idk what these are but they were in the demo
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
    }
    #send request and recieve a response
    resp = requests.request('GET', url, headers=headers, params=params)

    # Gracefully handle errors
    if resp.status_code != 200:

        msg = f"Fuck \t Code:{resp.status_code} "

        if resp.status_code == 400:
            msg+= "\n" + resp.json()["message"]
        
        raise Exception(msg)

    if return_json :
        return resp.json()
    
    return resp
    

def read_data_from_api():
    print("reading from road traffic API")
    s = perf_counter()

    # The API gives the data to you in 'pages'
    # it also gives the url to the next page or None if there is no other page
    # this code fires off a request and just to get the url for the first page
    # then it repeatedly requests a pages reads the data and requests the next page
    page_data = []

    # send off a request for page one just to get the chain started
    url = 'https://roadtraffic.dft.gov.uk/api/average-annual-daily-flow'
    params = {
    'page[size]': 900,
    "page[number]":1,
    "filter[year]":'2024,2023,2022,2021,2019,2018', # just recent years
    'filter[region_id]':3, # Scotland
    }
    # don't even read the response initially, just get the url we landed at
    next_url = send_request(url, params=params, return_json=False).url

    while next_url is not None:
        response = send_request(url=next_url)
        next_url = response["next_page_url"]

        # now we read the info from response["data"]
        df = pd.DataFrame.from_dict(
                response["data"]
            )[["year", "local_authority_id", "pedal_cycles", "longitude", "latitude"]]
        
        page_data.append( df.apply(pd.to_numeric) )


    final_df = pd.concat(page_data, ignore_index = True)

    e = perf_counter()
    print(f"reading from road traffic API took {e-s:.0f} seconds with {final_df.shape[0]:,} rows")
    return final_df





if __name__ == "__main__":

    traffic = read_data_from_api()

    traffic = gpd.GeoDataFrame(traffic,
                           geometry=[Point(lon, lat) for lat, lon in traffic[["latitude", "longitude"]].to_numpy() ],
        crs="EPSG:4326"  # WGS84 Latitude/Longitude
    ).to_crs(epsg=3857)


    pois = pd.read_csv("edinburgh_pois.csv")
    pois = gpd.GeoDataFrame(pois, 
                        geometry=[ Point(lon, lat) for lat, lon in pois[['lat', 'lon']].to_numpy() ],
        crs="EPSG:4326"  # WGS84 Latitude/Longitude
    ).to_crs(epsg=3857)

    poi_box = pois.dissolve().convex_hull.buffer(1).iloc[0]
    traffic = traffic[ traffic.intersects(poi_box, align=False) ]

    locals = traffic["local_authority_id"].unique() 

    col_local, col_traffic = create_colours_for_locs_and_assignments(
        locals 
        , traffic["local_authority_id"].to_numpy()
    )


    _, ax = plt.subplots()
    for loc in locals:
        places = traffic["local_authority_id"] == loc
        traffic[places].plot(ax=ax,
                          color=col_traffic[places], alpha= .5
                          , label = loc)
    plt.legend()
    plt.title("Traffic junctions with bikes coloured by local authority")
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    plt.show()
