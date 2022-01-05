"""
(dis)charging of a battery energy storage system. Electricity prices per hour EP, 
energy production from solar panels PV, and energy demand Dem are considered over
the entire horizon (0-24h) to minimize total costs TC. Arbitrage should take place
as the battery is (dis)charged (Pbat_dis & Pbat_ch) to/from the grid 
(Pgrid_out & Pgrid_in) at the optimal moments. As opposed to most of the examples
online, the problem is not formulated as a state-space model, but mostly relies 
on exogenous data for price, consumption and production.

You can either write out all of your discrete equations yourself with 
m.options.IMODE=3 or else let Gekko manage the time dimension for you. When you 
include an objective or constraint, it applies them to all of the time points that 
you specify. With m.options.IMODE=6, there is no need to add the set indices in 
Gekko such as [t]. Here is a simplified model:
"""

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

m       = GEKKO()
# horizon
m.time  = np.linspace(0,4,5)
# data vectors
EP      = m.Param([0.1,0.05,0.2,0.25,0.15])
Dem     = m.Param([10,12,9,8,11])
PV      = m.Param([10,11,8,10,9])
# constants
bat_cap = 13.5
ch_eff  = 0.94
dis_eff = 0.94
# manipulated variables
Pbat_ch = m.MV(lb=0, ub=4)
Pbat_ch.DCOST   = 0
Pbat_ch.STATUS  = 1
Pbat_dis = m.MV(lb=0, ub=4)
Pbat_dis.DCOST  = 0
Pbat_dis.STATUS = 1
Pgrid_in = m.MV(lb=0, ub=5)    
Pgrid_in.DCOST  = 0
Pgrid_in.STATUS = 1
Pgrid_out = m.MV(lb=0, ub=5) 
Pgrid_out.DCOST  = 0
Pgrid_out.STATUS = 1
#State of Charge Battery
SoC = m.Var(value=0.5, lb=0.2, ub=1)
#Battery Balance
m.Equation(bat_cap * SoC.dt() == -dis_eff*Pbat_dis + ch_eff*Pbat_ch)
#Energy Balance
m.Equation(Dem + Pbat_ch + Pgrid_in == PV + Pbat_dis + Pgrid_out)
#Objective
m.Minimize(EP*Pgrid_in)
# sell power at 90% of purchase (in) price
m.Maximize(0.9*EP*Pgrid_out)
m.options.IMODE=6
m.options.NODES=3
m.options.SOLVER=3 
m.solve()

# ploting the results
plt.subplot(3,1,1)
plt.plot(m.time,SoC.value,'b--',label='State of Charge')
plt.ylabel('SoC')
plt.legend()
plt.subplot(3,1,2)
plt.plot(m.time,Dem.value,'r--',label='Demand')
plt.plot(m.time,PV.value,'k:',label='PV Production')
plt.legend()
plt.subplot(3,1,3)
plt.plot(m.time,Pbat_ch.value,'g--',label='Battery Charge')
plt.plot(m.time,Pbat_dis.value,'r:',label='Battery Discharge')
plt.plot(m.time,Pgrid_in.value,'k--',label='Grid Power In')
plt.plot(m.time,Pgrid_in.value,':',color='orange',label='Grid Power Out')
plt.ylabel('Power')
plt.legend()
plt.xlabel('Time')
plt.show()