import xpress as xp
import pandas as pd
import numpy as np
from time import perf_counter
from helper_functions import *
import platform

if platform.system() == "Windows":
    xp.init('C:/xpressmp//bin/xpauth.xpr')


def create_and_solve_model(desire, dist_mat, bike_max, cost_bike, cost_station, budget, dist_max=5 ):

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
        xp.Sum( demand[i]*bikes[i] for i in I )
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

    # xp.setOutputEnabled(True)
    print("Solving")
    solve_start = perf_counter()
    prob.solve()
    solve_end = perf_counter()
    print(f"Solved in {solve_end-solve_start:.2f} seconds with {demand.shape[0]:,f} variables")

    #mip gap 
    MIP_gap= get_MIP_gap(prob)
    print(f"{MIP_gap=:.2%}")

    # look at the solution
    solution  = pd.DataFrame({
        "build": [ int(i) for i in prob.getSolution(build) ],
        "bikes": [int(i) for i in prob.getSolution(bikes) ],
        "desire": desire
    })

    # return the pertient info that was not inputted
    return solution, MIP_gap, near


if __name__ == "__main__":
    #read in data here as np.arrays
    folder = path("good_example_data", "05 Nov 2025 03-50PM")
    # demand = np.load(path(folder, "demand.npy"))
    all_dists = np.load(path(folder, "all_dists.npy"))

    # that demand data is wonky so ill just make up data
    from scipy.stats import poisson
    demand = poisson.rvs(size = all_dists.shape[0], mu = 5, random_state = 2025)

    r = 15
    all_dists = all_dists[:r,:r]
    demand = demand[:r]


    sol, mip, near = create_and_solve_model(
        desire=demand, dist_mat=all_dists, bike_max=50,
        cost_bike=580, cost_station=20_000, budget=2_000_000,
        dist_max=5_000
    )
    print(sol)
    near = np.array(near, dtype=int)
    print(near)

