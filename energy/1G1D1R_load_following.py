"""
Benchmark I: Load Following
The first benchmark problem represents load following, a common scenario in grid systems.
The optimizer seeks to match demand and supply with fluctuating demand dynamics. 
A single generator with ramping constraints attempts to respond to a single load with 
perfect foresight. The generation and demand match initially, but the generator must ramp
in order to ensure this throughout the horizon while minimizing overproduction. There
is no energy storage in this application, every load demand is matched by generation.
Individual case studies include ramp rate constraints, power production, and energy storage
operation as design variables. Variables used in the benchmark problems are defined below.

Symbol	    Description
J           objective function
d	        demand
g	        generation
r	        ramp rate
e	        storage inventory
ein, eout	energy stored, recovered
R	        renewable source
sin, sout	slack variables for storage switching
η           storage efficiency
n	        number of generating units
i	        subscript indicates product i
"""

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,1,101)
m = GEKKO(remote=False); m.time=t

d = m.Param(np.cos(2*np.pi*t)+3)
g = m.Var(d[0])
J  = m.CV(0)
J.STATUS=1; J.SPHI=J.SPLO=0
J.WSPHI=1000; J.WSPLO=1
r = m.MV(0,lb=-1,ub=1); r.STATUS=1
m.Equations([g.dt()==r, J==d-g])
m.options.IMODE=6; m.solve()

plt.plot(t,g,'b:',label='Production')
plt.plot(t,d,'r-',label='Demand')
plt.plot(t,r,'k--',label='Ramp Rate')
plt.legend(); plt.grid(); plt.show()