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

""" m.time  = np.linspace(0,24,25)
# data vectors
EP      = m.Param([0.1,0.05,0.2,0.25,0.15,
                   0.2,0.25,0.3,0.35,0.25,
                   0.25,0.3,0.25,0.25,0.2,
                   0.25,0.3,0.35,0.4,0.45,
                   0.4,0.35,0.3,0.25,0.2])
Dem     = m.Param([0, 0.772599,0.680559,0.647978,0.639773,0.647299,
                   0.701176,0.898463,1.18119,1.12899,0.980107,
                   0.974842,0.938832,0.89839,0.857997,0.84954,
                   0.910038,1.14956,1.60124,1.85407,1.8163,
                   1.70023,1.57341,1.30665,1.04025])

PV      = m.Param([0, -0.00116655,-0.00116655,-0.00116655,-0.00116655,
                   -0.00116655,-0.00116655,-0.00116655,0.259932,1.06851,
                   1.71611,0.605609,0.547296,1.55271,1.00728,0.315762,
                   0.266555,0.0289598,-0.00116655,-0.00116655,-0.00116655,
                   -0.00116655,-0.00116655,-0.00116655,-0.00116655]) """

m.time = np.linspace(0,24,25)
EP = m.Param([0,0.01874,0.01865,0.01892,0.01896,0.01837,
               0.02035,0.02082,0.02092,0.02156,0.02223,
               0.02229,0.02185,0.02116,0.02076,0.02058,
               0.02088,0.02413,0.02374,0.02304,0.02174,
               0.02088,0.02032,0.01999,0.01916])
    
PV = m.Param([0,-0.00116655, -0.00116655, -0.00116655, -0.00116655,
               -0.00116655, -0.00116655, -0.00116655, 0.0423505, 
                0.788561, 1.49915, 1.90253, 2.21281,
                2.32039, 2.11602, 1.17933, 0.554893,
               -0.00116655, -0.00116655, -0.00116655, -0.00116655,
               -0.00116655, -0.00116655, -0.00116655, -0.00116655])
    
Dem = m.Param([0,0.880622512, 0.765503361, 0.726264441, 0.721386598,
                0.726880416, 0.786228314, 1.010281023, 1.336666859,
                1.296280243, 1.118839407, 1.140846204, 1.125344746,
                1.08711696, 1.057013237, 1.053087139, 1.117596916,
                1.375524106, 1.855103218, 2.209266367, 2.146772546,
                1.986157285, 1.812116819, 1.486383131, 1.163033043])
    

# constants
bat_cap = 30
ch_eff  = 0.94
dis_eff = 0.94
# manipulated variables
Pbat_ch = m.MV(lb=0, ub=10)
Pbat_ch.DCOST   = 0
Pbat_ch.STATUS  = 1
Pbat_dis = m.MV(lb=0, ub=10)
Pbat_dis.DCOST  = 0
Pbat_dis.STATUS = 1
Pgrid_in = m.MV(lb=0, ub=7)    
Pgrid_in.DCOST  = 0
Pgrid_in.STATUS = 1
Pgrid_out = m.MV(lb=0, ub=7) 
Pgrid_out.DCOST  = 0
Pgrid_out.STATUS = 1
#State of Charge Battery
SoC = m.Var(value=0.5, lb=0.1, ub=1)
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
plt.plot(m.time,Dem.value,'r--',label='Load')
plt.plot(m.time,PV.value,'k:',label='PV')
plt.plot(m.time,Pgrid_in.value,'g:',label='Net Grid Demand')
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