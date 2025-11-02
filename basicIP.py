import xpress as xp
import pandas as pd
import numpy as np
from time import perf_counter
from helper_functions import *
# xp.init('C:/xpressmp//bin/xpauth.xpr')
prob = xp.problem("Most_Basic_Version") 

#read in data here as np.arrays
DATA_FOLDER = "good_example_data"
demand = np.load(path(DATA_FOLDER, "02 Nov 2025 10-02PM", "demand.npy"))
near = np.load(path(DATA_FOLDER, "02 Nov 2025 10-02PM", "near.npy"))


num_locs = demand.shape[0]

# we have the binary decision variable to build a station at location i
build = np.array(
    [prob.addVariable(name = f"build_{i}", vartype = xp.binary) for i in range(num_locs) ]
    ,dtype = xp.npvar
)

# objective function
prob.setObjective(
    xp.Sum( demand[i]*build[i] for i in range(num_locs) )
    , sense = xp.maximize
)

# constraints
# stay within budget
# thinking about it sum_i X_i*c <= B 
# <=> sum_i X_i <= floor(B/c)
# <=> sum_i X_i <= k, k is the maximum number of stations we can build
max_num_locations = max(num_locs//3, 1)#arbitary
prob.addConstraint(
    xp.Sum(build[i] for i in range(num_locs)) <= max_num_locations
)

######### new constraint: ensure that the stations are near each other #############

prob.addConstraint(
    build[i] <= xp.Sum( near[i,j]*build[j] for j in range(num_locs) if i !=j )
    for i in range(num_locs)
)


# Write problem statement to file, for debugging
prob.write("problem","ips")

# xp.setOutputEnabled(True)
print("Solving")
solve_start = perf_counter()
prob.solve()
solve_end = perf_counter()
print(f"Solved in {solve_end-solve_start:.2f} seconds with {demand.shape[0]:,.0f} variables")

#mip gap 
print(f"{get_MIP_gap(prob)=:.2f}")

# look at the solution
build_sol = pd.DataFrame(
    {"built": prob.getSolution(build) ,
     "demand": demand}
    , index=[ str(i) for i in build] )

print(build_sol[build_sol["built"] > 0])
print(prob.getSolution(build))
# print(near)





