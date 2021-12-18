"""
Benchmark IV: Constant Production with Storage
The fourth benchmark problem models a hybrid system with a single generator
with constant production constraints coupled with energy storage that together
must meet an oscillating electricity demand. The goal of the problem is to 
minimize the required power production and use energy storage to capture 
excess generation serve the oscillating energy demand while keeping the base-load
generator production constant. As before, the model has perfect foresight. 
In order to prevent the energy storage from charging and discharging simultaneously
without requiring mixed-integer variables, slack variables are used to control 
when the storage charges and discharges, allowing it to switch modes in a way 
that is both continuous and differentiable. This allows the modeling language 
to use automatic differentiation to provide exact gradients to the solver.
"""

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

m = GEKKO(remote=False)
m.time = np.linspace(0,1,101)

g = m.FV(); g.STATUS = 1 # production
s = m.Var(1e-2, lb=0)    # storage inventory
store = m.Var()          # store energy rate
s_in = m.Var(lb=0)       # store slack variable
recover = m.Var()        # recover energy rate
s_out = m.Var(lb=0)      # recover slack variable
eta = 0.7
d = m.Param(-2*np.sin(2*np.pi*m.time)+10)
m.periodic(s)
m.Equations([g + recover/eta - store >= d,
             g - d == s_out - s_in,
             store == g - d + s_in,
             recover == d - g + s_out,
             s.dt() == store - recover/eta,
             store * recover <= 0])
m.Minimize(g)

m.options.SOLVER   = 1
m.options.IMODE    = 6
m.options.NODES    = 3
m.solve()

plt.figure(figsize=(6,3))
plt.subplot(2,1,1)
plt.plot(m.time,d,'r-',label='Demand')
plt.plot(m.time,g,'b:',label='Prod')
plt.legend(); plt.grid(); plt.xlim([0,1])

plt.subplot(2,1,2)
plt.plot(m.time,s,'k-',label='Storage')
plt.plot(m.time,store,'g--', label='Store Rate')
plt.plot(m.time,recover,'b:', label='Recover Rate')
plt.legend(); plt.grid(); plt.xlim([0,1])
plt.show()