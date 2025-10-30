import xpress as xp
import pandas as pd
import numpy as np
xp.init('C:/xpressmp//bin/xpauth.xpr')
prob = xp.problem("Most_Basic_Version") 

#read in data here as np.arrays
locs = np.array([1,2,3,4])
demand = np.array([20, 20, 10, 15])
BUDGET = 3
cost = 1

# name our decision variables
num_locs = len(locs)
I = range(1,num_locs+1) # start counting from one

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
prob.addConstraint(
    xp.Sum(cost*build[i] for i in range(num_locs)) <= BUDGET
)

prob.write("problem","ip")

# xp.setOutputEnabled(True)
# prob.solve()
# print(build)





