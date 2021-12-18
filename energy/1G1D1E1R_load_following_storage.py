"""
Benchmark V: Load Following with Storage
The fifth benchmark combines energy storage with a load-following problem
similar to Benchmark I. The first-half of the time horizon is nearly 
identical to Benchmark I, but now the excess energy can be stored. This
allows the system to meet a higher demand in the second half of the time 
horizon without needing extremes in generation. The solver minimizes the 
ramping needs and operates more flexibly by storing and then recovering 
the overproduction caused by the ramping constraints. Energy storage allows
this generator to meet the load without requiring significant overproduction.
"""

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

m = GEKKO(remote=False)
m.time = np.linspace(0,1,101)

# renewable energy source
renewable = 3*np.cos(np.pi*m.time/6*24)+3
num = len(m.time)
center = np.ones(num)
center[0:int(num/4)] = 0
center[-int(num/4):] = 0
renewable *= center
r = m.Param(renewable)

dg = m.MV(0, lb=-4, ub=4); dg.STATUS = 1
d = m.Param(-2*np.sin(2*np.pi*m.time)+7)
net = m.Intermediate(d-r)
g = m.Var(d[0])       # production
s = m.Var(0, lb=0)    # storage inventory
store = m.Var()       # store energy rate
s_in = m.Var(lb=0)    # store slack variable
recover = m.Var()     # recover energy rate
s_out = m.Var(lb=0)   # recover slack variable
m.periodic(s)
eta = 0.85            # storage efficiency
m.Minimize(g)

err = m.CV(0); err.STATUS = 1
err.SPHI = err.SPLO = 0
err.WSPHI = 1000; err.WSPLO = 1
m.Minimize(0.01*err**2)

m.Equations([g.dt() == dg,  
             err == d - g - r + recover/eta - store,
             g + r - d == s_out - s_in,
             store == g + r - d + s_in,
             recover == d - g - r + s_out,
             s.dt() == store - recover/eta,
             store * recover <= 0])

m.options.SOLVER   = 1
m.options.IMODE    = 6
m.options.NODES    = 2
m.solve()

plt.figure(figsize=(7,5))
plt.subplot(3,1,1)
plt.plot(m.time,d,'r-',label='Demand')
plt.plot(m.time,g,'b:',label='Prod')
plt.plot(m.time,net,'k--',label='Net Demand')
plt.legend(); plt.grid(); plt.xlim([0,1])

plt.subplot(3,1,2)
plt.plot(m.time,r,'b-',label='Source')
plt.plot(m.time,dg,'k--',label='Ramp Rate')
plt.legend(); plt.grid(); plt.xlim([0,1])

plt.subplot(3,1,3)
plt.plot(m.time,s,'k-',label='Storage')
plt.plot(m.time,store,'g--', label='Store Rate')
plt.plot(m.time,recover,'b:', label='Recover Rate')
plt.xlim([0,1]); plt.xlabel('Time')
plt.legend(); plt.grid()
plt.show()
