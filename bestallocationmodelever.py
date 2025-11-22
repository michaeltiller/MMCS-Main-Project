import numpy as np
from helper_functions import *
import IP_models
from preprocessing import *
from demandfuncs import *
import pandas as pd 

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

old_demand = np.zeros(len(locations_gdf))
old_demand[locs_with_demand] = counts


traffic_based_demand = predict_bike_count_MLP(locations_gdf[["lat", "lon"]].to_numpy(), precomputed=True)

demand = (locations_gdf['prediced_Start_Trip_Counts'] * 0.4 + old_demand * 0.2)/365 + (traffic_based_demand/7)* .4

#### Get Distances #########
dist_mat = get_dists_gps(locations_gdf)

###### get train stations ######

train_stations = pd.read_csv('trainstations.csv')

# we want to reward building near a station - the reward is the additional daily trips
#    Edinburgh Waverley, Haymarket, Slateford, Wester Hailes, South Gyle,
#    Curriehill, Brunstane, Newcraighall, Gateway,Edinburgh business park
train_benefit = [80, 60, 40, 40, 30,
                 40, 40, 30, 40, 50 ]
# pred_train_benefit = predict_bike_count_MLP(train_stations[["Latitude", "Longitude"]].to_numpy(), precomputed=True)
# we really want the model to build near a station
# train_benefit = [ int(max(old, new/2)) for old, new in zip(train_benefit, pred_train_benefit)]

# define whether a location is near each train station
num_trains = train_stations.shape[0]
train_lon, train_lat = train_stations["Longitude"].to_numpy(), train_stations["Latitude"].to_numpy()

near_to_trains = np.zeros((num_trains, num_clusters), dtype = bool)
for t in range(num_trains):
    dist_to_train = haversine_np(
        train_lon[t], train_lat[t],
        loc_lon, loc_lat
        )
    near_to_trains[t] = dist_to_train <= .4 #400 metres
    print(f"Locs near {train_stations["Station"].iloc[t]} is {near_to_trains[t].sum()}")


################## Change Parameters here
BUDGET = 500_000 
BIKE_MAX = 30  # estimated from average of historical stations
COST_BIKE = 1_000 + 500 # cost + 1 year maintainence
COST_STATION = 5_000
DIST_MAX = 1
BIKES_TOTAL = 800

#### create a timestamp here for when we want to save any data later
stamp = timestamp()  + f" clusters-{num_clusters:,}"
save_folder = path("gen_data", stamp)
os.mkdir(save_folder)


print("Starting bike sensitivity to budget")
budget_param_vals = np.arange(start= 100_000, stop=2_000_000, step=100_000)
bikes_used = np.zeros(budget_param_vals.shape)
demand_met = np.zeros(budget_param_vals.shape)

for i, budget_param in enumerate(budget_param_vals):

    sol, mip, alloc_sol, train_sol  = IP_models.create_and_solve_extended_model(
    desire=demand, dist_mat=dist_mat,
    train_benefit=train_benefit,
    bike_max=BIKE_MAX, 
    cost_bike=COST_BIKE, 
    cost_station=COST_STATION, 
    budget=budget_param, # <----------
    near_trains=near_to_trains,
    dist_min = 0.4, #stations no closer than 400m
    dist_max =DIST_MAX,  #stations no more than 1km apart
    bikes_total=BIKES_TOTAL,
    verbose=False
    )
    x = summarise_solution(sol, train_sol, p_print=True)

    bikes_used[i] = x["bikes_used"]
    demand_met[i] = x["daily_demand_met"]

bikes_used_as_budget_varies = pd.DataFrame({
    "budget":budget_param_vals,
    "bikes_used":bikes_used,
    "daily_demand_met": demand_met
})
print(bikes_used_as_budget_varies)


title = "varying_budget_to_effect_demand_met_and_bikes_used"
_, ax = plt.subplots()
ax.plot(
    "budget", "daily_demand_met",
     label= "daily_demand_met",
     data= bikes_used_as_budget_varies,
     color = "red", linewidth = 2
)
ax.plot(
    "budget", "bikes_used",
     label = "bikes_used",
     data= bikes_used_as_budget_varies,
     color = "blue", linewidth = 2
)

ax.legend()
ax.set_title(title)
plt.savefig(path(save_folder, title+".pdf"))
plt.close()

bikes_used_as_budget_varies.to_csv( path(save_folder, title+".csv" )  )


bike_limits = np.arange(500, 1_000, 100)
budgets_used = np.zeros(bike_limits.shape)
b = 1_500_000
for i, bike_limit in enumerate(bike_limits):

    sol, mip, alloc_sol, train_sol  = IP_models.create_and_solve_extended_model(
        desire=demand, dist_mat=dist_mat,
        train_benefit=train_benefit,
        bike_max=BIKE_MAX, 
        cost_bike=COST_BIKE, 
        cost_station=COST_STATION, 
        budget= b, # <----------
        near_trains=near_to_trains,
        dist_min = 0.4, #stations no closer than 400m
        dist_max =DIST_MAX,  #stations no more than 1km apart
        bikes_total=bike_limit, # <---------
        verbose=False
    )
    x = summarise_solution(sol, train_sol, p_print=True)
    print(f"new {bike_limit=}, used:{ x["budget_used"]:,}  budget give:{b:,}")

    budgets_used[i] = x["budget_used"]

budget_used_as_bikes_vary = pd.DataFrame({
    "Budget_used" : budgets_used,
    "total_bikes" : bike_limits
})

print(budget_used_as_bikes_vary)


title = f"vary_total_bike_limit_to_effect_budget_used_of {b}"
_, ax = plt.subplots()
ax.plot(
    "total_bikes", "Budget_used",
     label= "Budget_used",
     data= budget_used_as_bikes_vary,
     color = "blue", linewidth = 2
)

ax.set_title(title)
plt.savefig(path(save_folder, title+".pdf"))
plt.close()

budget_used_as_bikes_vary.to_csv( path(save_folder, title+".csv" )  )



print("done bikes sensitivity to budget")



budget_for_phases = [500_000, 1_000_000,  2_000_000]
phases = len(budget_for_phases)
print(f"doing phases {budget_for_phases=}")
for phase, budget_param in enumerate(budget_for_phases):

    sol, mip, alloc_sol, train_sol  = IP_models.create_and_solve_extended_model(
    desire=demand, dist_mat=dist_mat,
    train_benefit=train_benefit,
    bike_max=BIKE_MAX, 
    cost_bike=COST_BIKE, 
    cost_station=COST_STATION, 
    budget=budget_param, # <----------
    near_trains=near_to_trains,
    dist_min = 0.4, #stations no closer than 400m
    dist_max =DIST_MAX,  #stations no more than 1km apart
    bikes_total=BIKES_TOTAL,
    verbose=False
    )
    summary = summarise_solution(sol, train_sol, COST_BIKE, COST_STATION)
    print(f"Budget given: {budget_param:,}  budget used: {summary["budget_used"]:,}")
    

    # Plotting and saving output for this phase
    m = make_map_from_sol(locations_gdf, train_stations,sol, alloc_sol, train_sol)
    # m.show_in_browser()

    tag = f"phase-{phase+1} budget-{budget_param} "
    m.save(path(save_folder, tag+"the_map.html") )
    sol.to_csv(path(save_folder, tag+"the_sol.csv"))

print("done phases stuff")




