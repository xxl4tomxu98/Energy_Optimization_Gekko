"""
The 5-weeks harvest period (#horizon: T=5) is upon us, and my produce will be: 
[3.0, 7.0, 9.0, 5.0, 4.0] Some apples I keep for myself [2.0, 4.0, 2.0, 4.0, 2.0], 
the remaining produce I will sell in the farmer's market at the following prices: 
[0.8, 0.9, 0.5, 1.2, 1.5]. I have storage space with room for 6 apples, 
so I can plan ahead and sell apples at the most optimal moments, hence maximizing 
my revenue. I try to determine the optimal schedule with the following model:

    Set a small penalty on moving inventory into or out of storage. Otherwise 
    the solver can find large arbitrary values with storage_in = storage_out.
    I used m.Minimize(1e-6*storage_in) and m.Minimize(1e-6*storage_out).

    Because the initial condition is typically fixed, I used zero values at the 
    beginning just to make sure that the first point is calculated.

    I also switched to integer variables if they are sold and stored in integer 
    units. You need to switch to the APOPT solver if you want an integer solution with SOLVER=1.
"""

from gekko import GEKKO
import numpy as np

m       = GEKKO(remote=False)
m.time  = np.linspace(0,5,6)
orchard   = m.Param([0.0, 3.0, 7.0, 9.0, 5.0, 4.0])
demand    = m.Param([0.0, 2.0, 4.0, 2.0, 4.0, 2.0]) 
price     = m.Param([0.0, 0.8, 0.9, 0.5, 1.2, 1.5])

### manipulated variables
# selling on the market
sell                = m.MV(lb=0, integer=True)
sell.DCOST          = 0
sell.STATUS         = 1
# saving apples
storage_out         = m.MV(value=0, lb=0, integer=True)
storage_out.DCOST   = 0      
storage_out.STATUS  = 1 
storage_in          = m.MV(lb=0, integer=True)
storage_in.DCOST    = 0
storage_in.STATUS   = 1

### storage space 
storage         = m.Var(lb=0, ub=6, integer=True)
### constraints
# storage change
m.Equation(storage.dt() == storage_in - storage_out) 

# balance equation
m.Equation(sell + storage_in + demand == storage_out + orchard)

# Objective: argmax sum(sell[t]*price[t]) for t in [0,4]
m.Maximize(sell*price)
m.Minimize(1e-6 * storage_in)
m.Minimize(1e-6 * storage_out)
m.options.IMODE=6
m.options.NODES=2
m.options.SOLVER=1
m.options.MAX_ITER=1000
m.solve()

print('Sell')
print(sell.value)
print('Storage Out')
print(storage_out.value)
print('Storage In')
print(storage_in.value)
print('Storage')
print(storage.value)