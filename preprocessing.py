from sklearn.cluster import KMeans
from shapely.geometry import MultiPoint, Point
from geopy.distance import great_circle
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from helper_functions import *
import  glob, os, threading, time
from scipy.spatial import cKDTree
import contextily as ctx
from time import perf_counter
import requests, ujson
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def designate_weight(cat):
    """designate a POI's weight based off it's category.
    
    Weight must be positive"""
    if cat in ("university", "commercial"):
        return 3
    elif cat in ("school", "hospital"):
        return 2
    elif cat in ("residential",  "library"):
        return 1
    else:
        return 1

def cluster_and_get_centremost_points(arr, num_clusters, kmeans_weights=None):
    """Takes in an array ``arr`` of latitude and longitude, in that order,  points.
       Performs kmeans clustering with weights ``kmeans_weights``.
       Then returns the 'centremost' point from each cluster
       The output is a GeoPandas dataframe"""
    
    if kmeans_weights is None:
        kmeans_weights = np.ones(arr.shape[0])
    
    s = perf_counter()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(arr, sample_weight=kmeans_weights)
    cluster_labels = kmeans.labels_


    clusters = pd.Series([arr[cluster_labels == n] for n in range(num_clusters)])

    def get_centermost_point(cluster):
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
        return tuple(centermost_point)

    centermost_points = clusters.map(get_centermost_point)
    e = perf_counter()
    print(f"clustering and getting centremost points took {e-s:.0f} seconds")

    # # Convert to GeoDataFrame to plot a picture 
    return gpd.GeoDataFrame(
        {
        "lat":centermost_points.map(lambda tup : tup[0]), 
        "lon":centermost_points.map(lambda tup : tup[1])
        }, 
        geometry=[Point(lon, lat) for lat, lon in centermost_points],
        crs="EPSG:4326"  # WGS84 Latitude/Longitude
    ).to_crs(epsg=3857)

########### find which locations are "close" to each other #####################
# we will use the osrm module for this 
# it has a web API which takes a url with the longatudes and latitudes 
# and returns a distance table
def dist_from_point_basic(p, arr):
    """get the distance from a single point ``p`` to each point in ``arr``.
       ``arr`` must be small enough to pass into the url string for osrm
    """

    url = "http://router.project-osrm.org/table/v1/bike/"
    #add p as the first coordinates
    url += f"{p[1]},{p[0]};"
    #its longitude first  <------------------
    url += ";".join( f"{lon},{lat}" for lat, lon in arr )
    url += "?sources=0&annotations=distance"
    # time.sleep(2)
    # t = datetime.datetime.now()
    # print(f"requesting at {t} ") # the f strings lazy evaluate
    resp = requests.get(url)

    if resp.status_code != 200:
        print("fuck " + resp.text) 
        return np.inf

    # stored as a list within a list because there is only one source point
    dists = ujson.loads(resp.content)["distances"][0]
    # An annoyance is that in each chunk the first value is the dist from p to p
    # so trim that out 
    # this is only because p is added into the front of the array every time so it keeps calculating that
    return np.array(dists[1:])


def get_distances(coords_arr):
    n_rows = coords_arr.shape[0]
    dists = np.zeros((n_rows,n_rows))

    #the we can only do so many co-ordinates at once -> break it up into batches
    MAX_BATCH_SIZE = 325 #works, 330 is too large, try rise it further if you want
    num_batches = np.ceil(n_rows/MAX_BATCH_SIZE)
    # split the array into a list of num_batches arrays
    # each small array has at most MAX_BATCH_SIZE elements
    batched = np.array_split(coords_arr, num_batches)

    
    # i was trying to doing something more effecient than this using threading
    # but i got blocked, I believe for sending more than one request per minute
    print("On:")
    for i in range(n_rows):
        print(i)
        dists[i,] = np.concat( [ dist_from_point_basic(coords_arr[i], batch) for batch in batched ] )
    print()
    return dists


def get_distances_threaded(coords_arr):
    """Doesn't work"""
    s = perf_counter()
    n_rows = coords_arr.shape[0]
    MAX_BATCH_SIZE = 325 #works, 330 is too large, try rise it further if you want
    
    col_inds_split = np.split(
        np.arange(n_rows),
        range(MAX_BATCH_SIZE, n_rows, MAX_BATCH_SIZE)
    )

    dists = np.zeros((n_rows,n_rows))
    threads = []

    def dist_from_point_batched(row_i,col_is):
        dists[row_i, col_is] = dist_from_point_basic(coords_arr[row_i], coords_arr[col_is])

    for row_ind in range(n_rows):
        for col_inds in col_inds_split:
            t = threading.Thread(
                target=dist_from_point_batched,
                args=(row_ind, col_inds) 
            )
            threads.append(t)  
    
    for t in threads:
        t.start()
        time.sleep(3) #1 request per second

    for t in threads:
        t.join()


    e = perf_counter()
    print(f"getting dists threaded took {e-s:.0f} seconds")
    return dists



def get_historical_trips(and_end_points = False):
    """ get the demand for each potential location.
    returns it as a 2D array of longitude and latitude 
     """
    s = perf_counter()
    # # read the trip data from csv into dataframes
    filenames = glob.glob("*.csv", root_dir="cyclehire-cleandata") #regex
    trips = pd.concat([ pd.read_csv(path("cyclehire-cleandata", f )) for f in filenames  ])

    # # we consider a start location and an end location as two different "units" of demand
    # # i.e. there is a demand of one wherever the trip started and a demand of one wherever the trip ended
    col_names = ["lon","lat"]
    starts = trips[['start_station_longitude', 'start_station_latitude' ]]
    starts.columns = col_names

    if and_end_points:
        ends = trips[['end_station_longitude','end_station_latitude' ]]
        ends.columns = col_names
        demand = pd.concat([starts,ends], ignore_index=True)
    else:
        demand = starts
    
    df = gpd.GeoDataFrame(
        {
        "lon": demand["lon"],
        "lat": demand["lat"]
        },
        geometry=[Point(tup) for tup in zip(demand["lon"], demand["lat"])],
        crs="EPSG:4326"  # WGS84 Latitude/Longitude
    ).to_crs(epsg=3857)

    e = perf_counter()
    print(f"reading historical data took {e-s:.0f} seconds")

    return df

def assign_points_to_nearest_location(points, locs):
    """
    Does what it says on the tin. 
    Output is an array ``arr`` 
    where ``arr[i]`` gives the index of the location that point ``i`` was assigned to

    Ensure both ``points`` and ``locs`` have columns ``lon```and ``lat``.
    """
    points = points[["lon", "lat"]]
    locs = locs[["lon", "lat"]]

    s = perf_counter()

    dist_tree = cKDTree(locs)
    _, nearest_loc = dist_tree.query(points, k = 1)

    e = perf_counter()
    print(f"Assigning points to locations took {e-s:.0f} seconds")

    return nearest_loc

def show_elbow_of_weighted_kmeans(locs, k_values = np.arange(1,10+1), loc_weights = None):
    """
    Performs weighted k means repeatedly on ``locs`` using each value in ``k_values`` as the number of 
    clusters, k.
    Plots and returns the SS dists from the centroids, also known as the entropy, for each choice of k.

    """
    s = perf_counter()
    
    k_values = np.array(k_values)

    # Making this into a numpy ufunc so it _might_ run faster
    get_entropy  = np.vectorize(
        lambda k: KMeans(k, random_state=0).fit(locs, sample_weight=loc_weights).inertia_
        )
    
    entropies = get_entropy(k_values)

    e = perf_counter()
    print(f"Getting elbow plot took {e-s:.0f} seconds for {len(k_values)} values of k")

    _, ax = plt.subplots()
    ax.plot(k_values, entropies, "b-")
    ax.scatter(k_values, entropies, c="red")
    ax.set_xlabel("K")
    plt.title("Sum of square dists to centroid")
    plt.show()

    return entropies


def location_score(locs:gpd.GeoDataFrame):
    """
    Implements the L score for a location
    """
    # GH will give us more stuff for here
    z_cat = locs["cat"].apply(designate_weight)

    return z_cat

def distribute_regional_demand_by_L_score(regions, locations):
    """
    takes in a ``regions`` data frame with a demand and L_score and a 
    ``locations`` data frame with a region and L_score column. 

    It assigns regional demand to each location proportional to the location's L_score over the regional L_score.
    """
    # location score is positive so a region's total is positive 
    demand_per_l_score_unit = regions["demand"] / regions["L_score"]
    return np.ceil( demand_per_l_score_unit[locations["region"]] * locations["L_score"] )


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance, in km, between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length. 

    Source: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km


def get_dists_gps(df, memory_intense = False ):
    """
    gets the distance matrix, using gps distance, of the points given by the ``lat`` and ``lon`` columns of ``df``.

    Should take ~ 1 min for 33K.

    The option ``memory_intense`` may or may not be more time-efficient but seems to max out my 8GB of RAM.
    """
    print("getting gps dists")
    s = perf_counter()
    if memory_intense:
        lon_lat= df[["lon","lat"]]

        def wrapper(tup, **kwargs):
            return haversine_np(*tup, **kwargs )

        result =  lon_lat.apply(
            func = wrapper, lat2 =df["lat"].values, lon2 = df["lon"].values, axis = 1
        ).to_numpy()
    
    else:
        d = np.zeros( (df.shape[0], df.shape[0]), dtype=np.float16 )
        for i, (lon, lat) in enumerate(zip(df["lon"], df["lat"])):
            d[i,] =  haversine_np(lon, lat, df["lon"], df["lat"])
        result = d
    
    e= perf_counter()
    print(f"getting gps dists took {e-s:.0f} secs")

    return result


def read_data_from_api():
    """
    Read bike count data at traffic junctions 
    
    See  https://roadtraffic.dft.gov.uk/docs/index.html 
    """
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
    'page[size]': 30_000, # keep increasing this until something breaks
    "page[number]":1,
    # "filter[year]":'2024'
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
            )

        df = df[["year", "local_authority_id", "pedal_cycles", "longitude", "latitude"]]
        page_data.append( df.apply(pd.to_numeric) )


    final_df = pd.concat(page_data, ignore_index = True)

    e = perf_counter()
    print(f"reading from road traffic API took {e-s:.0f} seconds with {final_df.shape[0]:,} rows")
    return final_df


def read_traffic_data():
    """
    get bike traffic at junctions in edinburgh
    """
    #get all traffic junction data in edinburgh
    traffic = read_data_from_api()

    # use only recent years
    traffic = traffic[ 
        (traffic["year"] != 2021) &
        (traffic["year"] != 2020) &
        (traffic["year"] >= 2018) &
        (traffic["year"] <= 2024) 
    ]

    # make it into a Geodataframe
    traffic = gpd.GeoDataFrame(traffic,
                           geometry=[Point(lon, lat) for lat, lon in traffic[["latitude", "longitude"]].to_numpy() ],
        crs="EPSG:4326"  # WGS84 Latitude/Longitude
    ).to_crs(epsg=3857)

    # we only care about the junctions near where the stations might be
    pois = pd.read_csv("edinburgh_pois.csv")
    pois = gpd.GeoDataFrame(pois, 
                        geometry=[ Point(lon, lat) for lat, lon in pois[['lat', 'lon']].to_numpy() ],
        crs="EPSG:4326"  # WGS84 Latitude/Longitude
    ).to_crs(epsg=3857)
    
    # get the area covered by the POIs plus a 1km buffer i.e the city of Edinburgh 
    poi_box = pois.dissolve().convex_hull.buffer(1).iloc[0]

    # consider only the junctions in Edinburgh
    traffic = traffic[ traffic.intersects(poi_box, align=False) ]

    return traffic


def predict_bike_count_MLP( new_x, show_plot=False):
    """
    This predicts bike count data from the longitude and latitude of ``new_x``.
    It does this by looking at recent historical bike traffic at junctions in Edinburgh.
    The model used was a MultiLayer Perceptron - a simple neural network.

    Right now the prediction is not impressive but I think we have reached the point of diminishing returns.
    """

    # traffic = read_traffic_data()
    traffic = pd.read_csv("edinburgh_traffic_data_2018-2024_no_2020-2021.csv")

    train_X = traffic[["latitude", "longitude"]].to_numpy()
    train_y =  traffic["pedal_cycles"].to_numpy()

    # scaling the data improved performance
    scaler = StandardScaler()
    train_X = scaler.fit(train_X).transform(train_X)

    print(f"Using {train_X.shape[0]:,} observations to predict {new_x.shape[0]:,} outcomes")


    def plot_preds_against_gps( y_pred, title=""):
        """Plot predictions on training data against training responses"""
        lat, lon = traffic["latitude"], traffic["latitude"]
        y  = traffic["pedal_cycles"]
        _, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.scatter(lat, y, color="black")
        ax2.scatter(lon, y, color="black")
        
        ax1.scatter(lat, y_pred, color="red")
        ax2.scatter(lon, y_pred, color="red")

        plt.title(title + " training predictions")
        plt.show()

    mlp_mod = MLPRegressor(hidden_layer_sizes=(75, 50), solver='lbfgs',
                        activation='relu', max_iter= 10_000)
    mlp_mod.fit(train_X, train_y)

    if show_plot:
        plot_preds_against_gps(mlp_mod.predict(train_X), "basic MLP")


    # other approaches I tried was poisson regression and a random NN
    # import sklearn
    # poisson_mod = sklearn.linear_model.PoissonRegressor()
    # poisson_mod.fit(train_X, train_y)
    # plot_preds_against_gps( poisson_mod.predict( train_X), "poisson")

    # import keras
    # nn2 = keras.Sequential([
    #         keras.layers.Dense(100, activation="relu"),
    #         keras.layers.Dense(100, activation="relu"),
    #         keras.layers.Dense(10, activation="relu"),
    #         keras.layers.Dense(1),
    #     ])
    # nn2.compile(loss="mse")
    # nn2.fit(x=train_X,y=train_y, epochs = 100, verbose=0)
    # plot_preds_against_gps(nn2.predict(train_X), "keras nn")


    # ensure that the new data is on the scale the model expects
    new_y_pred = mlp_mod.predict( scaler.transform(new_x) )

    return new_y_pred

if __name__ == "__main__":

    pois = pd.read_csv("edinburgh_pois.csv")

    poi_weights = pois['category'].apply(designate_weight)
    coords = pois[['lat', 'lon']].to_numpy()
    # print(coords.shape)
    # sum_sq = show_elbow_of_weighted_kmeans(coords, range(1, 50+1), poi_weights)
    # locations_gdf = cluster_and_get_centremost_points(coords, 9, poi_weights)

    # hist_starts_gdf = get_historical_trips()


    # r = list(range(30,60))
    # locations_gdf = locations_gdf.iloc[r]
    # # hist_starts = hist_starts.iloc[r]


    # near_labs = assign_points_to_nearest_location(hist_starts_gdf, locations_gdf)
    # locs_with_demand, counts = np.unique(near_labs, return_counts = True)
    # # print(u)
    # print(len(counts), counts)
    # colour_map = plt.colormaps['cool'].resampled( min(len(r), 10) )
    # cols = colour_map(range(locations_gdf.shape[0]))


    # # f, (ax1, ax2) = plt.subplots(ncols=2)
    # _, ax1 = plt.subplots()

    # hist_starts_gdf.to_crs(locations_gdf.crs).plot(
    #     ax=ax1, label="Historical",
    #     color=cols[near_labs],
    #     markersize=10
    #     ,marker="."
    # )
    
    # locations_gdf.plot(
    #     ax=ax1, label="Locations",
    #     color=cols,
    #     markersize=30,
    #     marker = "x"
    # )
    
    # ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
    # ax1.legend()
    # plt.title("Colours show assignment")

    # # hist_starts.to_crs(rep_points_gdf.crs).plot(
    # #     ax=ax2, label="Historical",
    # #     color=cols[near_labs],
    # #     markersize=20
    # #     ,marker="D"
    # # )
    # # ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
    # # ax2.legend()


    # plt.show()
    # this gets you blocked
    # locs_as_tuples = locations_gdf.apply(lambda row: (row["lat"], row["lon"]), axis=1 ).to_numpy()
    # print(f"{locs_as_tuples.shape=}")
    # d = get_distances_threaded(locs_as_tuples)

    # print(d)

    # locations_gdf = cluster_and_get_centremost_points(coords, 9, poi_weights)["geometry"]
    # num_regions = 10

    # locations_gdf = gpd.GeoDataFrame({
    #     "region": KMeans(n_clusters=num_regions, random_state=0).fit(coords, sample_weight=poi_weights).labels_ ,
    #     "lat": coords[:,0],
    #     "lon": coords[:,1]
    #     },
    #     geometry=[Point(lon, lat) for lat, lon in coords],
    #     crs="EPSG:4326"  # WGS84 Latitude/Longitude
    # ).to_crs(epsg=3857)

    # col_regions, col_region_points = create_colours_for_locs_and_assignments(
    #     np.arange(num_regions), locations_gdf["region"]
    # )


    # regions_gdf = locations_gdf.dissolve(by="region")

    # _, ax = plt.subplots()
    # regions_gdf.convex_hull.plot(ax=ax,
    #                             alpha = .2, color=col_regions)
    # regions_gdf.convex_hull.boundary.plot(ax=ax, color="black")
    # regions_gdf.plot(ax=ax, color = col_region_points)
    # ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    # plt.show()

    # locations_gdf = gpd.GeoDataFrame({
    #     "name":pois["name"],
    #     # "region": KMeans(n_clusters=num_regions, random_state=0).fit(coords, sample_weight=poi_weights).labels_ ,
    #     "lat": coords[:,0],
    #     "lon": coords[:,1],
    #     "cat": pois["category"]
    #     },
    #     geometry=[Point(lon, lat) for lat, lon in coords],
    #     crs="EPSG:4326"  # WGS84 Latitude/Longitude
    # ).to_crs(epsg=3857)

    # l = locations_gdf.iloc[0:3]
    # print(l.shape)
    # s = perf_counter()
    # dd= get_dists_gps(l)
    # e = perf_counter()
    # print(f"get_dists_gps took {e-s:.0f} secs")
    # print(dd)

    ### traffic data demo

    traffic = read_traffic_data()

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

