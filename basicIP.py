import xpress as xp
import pandas as pd
import numpy as np
from time import perf_counter
from helper_functions import *
# xp.init('C:/xpressmp//bin/xpauth.xpr')
prob = xp.problem("Most_Basic_Version") 

#read in data here as np.arrays
DATA_FOLDER = "gen_data"
demand = np.load(path(DATA_FOLDER, "31 Oct 2025 10-28AM", "demand.npy"))
locs = np.array(range(demand.shape[0])) #no names for now, this can be changed later
# thinking about it sum_i X_i*c <= B 
# <=> sum_i X_i <= floor(B/c)
# <=> sum_i X_i <= k, k is the maximum number of stations we can build
max_num_locations = len(locs)//3 #arbitary

# name our decision variables
num_locs = len(locs)

# we have the binary decision variable to build a station at location i
build = np.array(
    [prob.addVariable(name = f"build_{i}", vartype = xp.binary) for i in locs ]
    ,dtype = xp.npvar
)

# objective function
prob.setObjective(
    xp.Sum( demand[i]*build[i] for i in range(num_locs) )
    , sense = xp.maximize
)

# constraints
# stay within budget
prob.addConstraint(
    xp.Sum(build[i] for i in range(num_locs)) <= max_num_locations
)
# Write problem statement to file, for debugging
prob.write("problem","ips")

# xp.setOutputEnabled(True)
print("Solving")
solve_start = perf_counter()
prob.solve()
solve_end = perf_counter()
solve_time = solve_end-solve_start
print(f"Solved in {solve_time:.2f} seconds with {demand.shape[0]:,.0f} variables")
#mip gap 
print(f"{get_MIP_gap(prob)=:.2f}")
# look at the solution
# build_sol = pd.DataFrame(
#     {"built": [prob.getSolution(i) for i in build],
#      "demand": demand}
#     , index=[ str(i) for i in build] )
# print(build_sol)





