import xpress as xp
import pandas as pd
import numpy as np
from shapely import Point
from time import perf_counter
from helper_functions import *
import platform
from itertools import combinations

if platform.system() == "Windows":
    xp.init('C:/xpressmp//bin/xpauth.xpr')


def create_and_solve_first_model(desire, dist_mat, bike_max, cost_bike, cost_station, budget, dist_max=5_000 ):

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
        xp.Sum( desire[i]*bikes[i] for i in I )
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

    # Every station must have at least one near it
    near = dist_mat <= dist_max
    prob.addConstraint(
        build[i] <= xp.Sum( near[i,j]*build[j] for j in I if j != i)
        for i in I
    )

    # stay within budget
    prob.addConstraint(
        xp.Sum( cost_bike*bikes[i] + cost_station*build[i] for i in I) <= budget
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

    # return the pertient info that was not inputted
    return solution, MIP_gap, near



def create_and_solve_basic_distmin_model(desire, dist_mat, near_centre, cost_bike, cost_station, budget,
                                     dist_max=5, dist_min_in_centre = .250, dist_min_outside_centre =.750,
                                      bikes_max = 50, bikes_total = 800 ):

    print("setting up in IP")
    s= perf_counter()
    # stop the big stream of text
    # xp.setOutputEnabled(False)
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
        xp.Sum( desire[i]*bikes[i] for i in I )
        , sense = xp.maximize
    )

    ########### constraints #############

    # we can place at most bikes_max bikes in each location
    # and if we put bikes somewhere then we must build a station there
    prob.addConstraint(
        bikes[i] <= bikes_max*build[i] for i in I
    )

    # we only have bikes_total bikes
    prob.addConstraint(
        xp.Sum(bikes[i] for i in I) <= bikes_total
    )

    # If we build a station somewhere then we must put at least one bike there
    prob.addConstraint(
        build[i] <= bikes[i] for i in I
    )

    # Every station must have at least one near it
    near = dist_mat <= dist_max
    prob.addConstraint(
        build[i] <= xp.Sum( build[j] for j in I if j != i and near[i,j])
        for i in I
    )

    # stations cannot be too close together
    # there is a different cut-off for too close, inside and outside of the city centre
    too_close = np.zeros((num_locs, num_locs), dtype=np.bool)
    for i in I:
        if near_centre[i]:
            cutoff = dist_min_in_centre
        else:
            cutoff = dist_min_outside_centre
        new = dist_mat[i] <= cutoff
        too_close[i] = new

    prob.addConstraint(
        build[i] + xp.Sum( build[j] for j in I if j !=i and too_close[i,j]) <= 1
        for i in I
    )

    # stay within budget
    prob.addConstraint(
        xp.Sum( cost_bike*bikes[i] + cost_station*build[i] for i in I) <= budget
    )

    ########## Solving ###########
    # Write problem statement to file, for debugging
    # prob.write("problem","ips")
    del too_close, near, dist_mat, near_centre
    e = perf_counter()
    print(f"set-up in IP took {e-s:.0f} seconds")

    # attr = prob.attributes.__dict__
    # for key in attr:
    #     if "memory" in key:
    #         print(f"{key} {attr[key]}")

    print("Solving")
    solve_start = perf_counter()
    prob.solve()
    solve_end = perf_counter()
    print(f"Solved in {solve_end-solve_start:.0f} seconds with {desire.shape[0]:,} variables")

    #mip gap 
    MIP_gap= get_MIP_gap(prob)
    print(f"{MIP_gap=:.2%}")

    # attr = prob.attributes.__dict__
    # for key in attr:
    #     if "memory" in key:
    #         print(f"{key} {attr[key]}")

    # look at the solution
    solution  = pd.DataFrame({
        "build": np.array([ int(i) for i in prob.getSolution(build) ]),
        "bikes": np.array([int(i) for i in prob.getSolution(bikes) ]),
        "desire": desire
    })



    # return the pertient info that was not inputted
    return solution, MIP_gap



def create_and_solve_extended_model(desire, dist_mat, bike_max, 
                                    cost_bike, cost_station, budget,
                                    rev_per_bike, near_to_trains,
                                    dist_min =100, dist_max=5_000):

    # stop the big stream of text
    xp.setOutputEnabled(False)
    prob = xp.problem("First_bike_extension") 

    num_locs = desire.shape[0]
    I = set(range(num_locs))
    Trains = range(near_to_trains.shape[0])

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
    
    
    allocated = np.array(
        [[prob.addVariable(name=f'allocated_{i}_{j}', vartype=xp.binary) for j in I]
        for i in I]
    )
    
    # we may choose to build near / cover a train station
    train_covered = np.array(
        [prob.addVariable(name=f"covered_train_{t}", vartype=xp.binary) for t in Trains]
        , dtype=xp.npvar
    )
    gamma = 10_000  # reward for covering a train station



    ########### Objective function #############
    # prob.setObjective(
    #     xp.Sum(desire[i] * bikes[i] for i in I) + gamma * xp.Sum(train_covered[t] for t in Trains),
    #     sense=xp.maximize
    # )


    ########### Soft train-station coverage #############
    # we can cover a train station by building near it
    for t in Trains:
        nearby_locs = np.where(near_to_trains[t])[0]
        if nearby_locs.size > 0:
            prob.addConstraint(train_covered[t] <= xp.Sum(build[i] for i in nearby_locs))
        else:
            #if there are no stations near it we can never cover it
            prob.addConstraint(train_covered[t] == 0)

    
    
    # if u want it to run fast use the below objective function 
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

    # we have a total of 800 bikes
    prob.addConstraint(
        xp.Sum( bikes[i] for i in I) <= 800
    )

        
    # stay within budget
    prob.addConstraint(
        xp.Sum( cost_bike*bikes[i] + cost_station*build[i] for i in I) <= budget
    )
    
    # Basic min Distance constraints #####
    
    too_close = np.zeros_like(dist_mat, dtype=int)
    for i in I:
        for j in I:
            if i < j:
                if dist_mat[i,j] < dist_min:
                    too_close[i,j] = 1
                    too_close[j,i] = 1

    
    prob.addConstraint(build[i] + xp.Sum(too_close[i,j]*build[j] for j in I if j != i) <= 1 for i in I)


    ## connectivity Constraints ####
    
    prob.addConstraint(allocated[i,j] <= build[i] for i in I for j in I)
    prob.addConstraint(allocated[i,j] <= build[j] for i in I for j in I)
    
    
    prob.addConstraint(
        build[j] == xp.Sum(allocated[j, i] for i in I) 
        for j in I 
    )
    
    # this is what fucked it - having the the below constraint prevents the spoke style connections 
    # prob.addConstraint(
    #     build[j] == xp.Sum(allocated[i,j] for i in I) 
    #     for j in I 
    # )
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
 
    print("Solving first without connectedness")
    solve_start = perf_counter()
    prob.solve()
    solve_end = perf_counter()
    print(f"Solved in {solve_end-solve_start:.0f} seconds with {desire.shape[0]:,} variables")

    #mip gap 
    MIP_gap= get_MIP_gap(prob)
    print(f"{MIP_gap=:.2%}")

    

    temp = prob.getSolution(allocated)

    ######### Cycling contraint ##########
    arcs = []

    for i in I:
        row_i = temp[i]
        j = np.where(row_i)[0]
        if j.any():
            arcs.append((i,j[0].item()))
    
    # size of subsets where we do not want a loop
    size_s = 3
    
    
   
    print("adding connectedness constraints")
    s_connect=perf_counter()
    #I suspect that in the second solution we build in areas we hadn't before 
    # - areas not subject to the below constraints
    # hence we see cycles

    originally_built = prob.getSolution(build).nonzero()[0]
    print(f" {len(originally_built)=:,} ")

    all_built_subsets = combinations(originally_built , size_s)
    
    # each village must have an arc inwards from outside the village
    # count = 0
    for s in all_built_subsets:

        outside_s = I.difference(set(s))
        prob.addConstraint(
            xp.Sum(allocated[i, j] for i in outside_s for j in s)
            <=
            xp.Sum( build[i] for i in s) / size_s
        )    
    #     count +=1

    # map(
    #     lambda s : prob.addConstraint(
    #         xp.Sum(allocated[i, j] for i in I.difference(set(s)) for j in s)
    #         <=
    #         xp.Sum( build[i] for i in s) / size_s
    #     )
    #     , all_built_subsets
    # )

    
    # no loops in a village / subset s
    prob.addConstraint(
        xp.Sum(allocated[i,j] for i in s for j in s )
        <= size_s - 1
        for s in all_built_subsets
    )
    e_connect = perf_counter()
    print(f"done adding connectedness {e_connect-s_connect:.0f} secs")
    # for s in all_subsets:
        
    #     s = all_subsets[0]
    #     notinS = [el for el in G.nodes() if el not in s]
    #     prob.addConstraint(
    #         xp.Sum(allocated[i, j] for i in notinS for j in s)
    #         <=
    #         xp.Sum( build[i] for i in s) / size_s
    #     )
        
    #     # no loops in a village
    #     prob.addConstraint(
    #         xp.Sum(allocated[i,j] for i in s for j in s )
    #         <= size_s - 1
    #     )
    
    

    print("Resolving")
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
    
    alloc_solution = np.zeros_like(allocated, dtype=bool)
    temp = prob.getSolution(allocated)
    
    
    for i in I:
        for j in I:
            alloc_solution[i, j] = bool(temp[i, j])
    del temp

    # Put into a DataFrame for readability
    alloc_df = pd.DataFrame(
        alloc_solution,
        columns=[f"to_{j}" for j in I],
        index=[f"from_{i}" for i in I]
    )
   
    
    # return the pertient info that was not inputted
    return solution, MIP_gap, alloc_df, arcs



if __name__ == "__main__":

    # ## look at the first model
    # #read in data here as np.arrays
    # folder = path("good_example_data", "05 Nov 2025 03-50PM")
    # # demand = np.load(path(folder, "demand.npy"))
    # all_dists = np.load(path(folder, "all_dists.npy"))

    # # that demand data is wonky so ill just make up data
    # from scipy.stats import poisson
    # demand = poisson.rvs(size = all_dists.shape[0], mu = 5, random_state = 2025)

    # r = 15
    # all_dists = all_dists[:r,:r]
    # demand = demand[:r]


    # sol, mip, near = create_and_solve_first_model(
    #     desire=demand, dist_mat=all_dists, bike_max=50,
    #     cost_bike=580, cost_station=20_000, budget=2_000_000,
    #     dist_max=5_000
    # )
    # print(sol)
    # near = np.array(near, dtype=int)
    # print(near)

    ### look at the model with min and max distances and bikes

    pass

