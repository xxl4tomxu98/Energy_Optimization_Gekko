"""
Objective: Understand the effect of bad data on dynamic optimization algorithms including 
estimator and control performance. Create a Python script to simulate and 
display the results. The flowrate of mud and cuttings is especially important with managed
pressure drilling (MPD) in order to detect gas influx or fluid losses. There are a range 
of measurement instruments for flow such as a mass flow meter or Coriolis flow meter 
(most accurate) and a paddle wheel (least accurate). This particular system has dynamics 
that are described by the following equation with Cv=1, u is the valve opening, 
and d is a disturbance.

 0.1 dF1/dt = -F1 + Cv u + d

Determine the effect of bad data (outliers, drift, and noise) on estimators such 
as moving horizon estimation. There is no need to design the estimators for this problem. 
The estimator scripts are below with sections that can be added to simulate the 
effect of bad data. Only an outlier has been added to these code. The code should be 
modified to include other common phenomena such as measurement drift 
(gradual ramp away from the true value) and an increase in noise (random fluctuations). 
Comment on the effect of corrupted data on real-time estimation and why some methods 
are more effective at rejecting bad data.
"""

from __future__ import division
from gekko import GEKKO
import numpy as np
import random

# intial parameters
n_iter = 150 # number of cycles
x = 37.727 # true value
# filtered bias update
alpha = 0.0951
# mhe tuning
horizon = 30

# Solve options
rmt = True # Remote: True or False

#Initialize model
m = GEKKO(remote=rmt)

# For rmt=True, specify server
m.server = 'https://byu.apmonitor.com'

#time array 
m.time = np.arange(50)

#Parameters
u = m.Param(value=42)
d = m.FV(value=0)
Cv = m.Param(value=1)
tau = m.Param(value=0.1)

#Variable
flow = m.CV(value=42)

#Equation 
m.Equation(tau * flow.dt() == -flow + Cv * u + d)

# Options
m.options.imode = 5
m.options.ev_type = 1 #start with l1 norm
m.options.coldstart = 1
m.options.solver = 1  # APOPT solver

d.status = 1
flow.fstatus = 1
flow.wmeas = 100
flow.wmodel = 0
#flow.dcost = 0

# Initialize L1 application
m.solve()

# Create storage for results
xtrue = x * np.ones(n_iter+1)
z = x * np.ones(n_iter+1)
time = np.zeros(n_iter+1)
xb = np.empty(n_iter+1)
x1mhe = np.empty(n_iter+1)
x2mhe = np.empty(n_iter+1)

# initial estimator values
x0 = 40
xb[0] = x0
x1mhe[0] = x0
x2mhe[0] = x0

# outliers
for i in range(n_iter+1):
    z[i] = x + (random.random()-0.5)*2.0
z[50] = 100
z[100] = 0


## Cycle through measurement sequentially
for k in range(1, n_iter+1):
    print( 'Cycle ' + str(k) + ' of ' + str(n_iter))
    time[k] = k

    # L1-norm MHE
    flow.meas = z[k] 
    m.solve()
    x1mhe[k] = flow.model

print("Finished L1")

#clear L1//
m.clear_data()
# Options for L2
m.options.ev_type = 2 #start with l1 norm
m.options.coldstart = 1 #reinitialize

flow.wmodel = 10

# Initialize L2 application
m.solve()

## Cycle through measurement sequentially
for k in range(1, n_iter+1):
    print ('Cycle ' + str(k) + ' of ' + str(n_iter))
    time[k] = k

    # L2-norm MHE
    flow.meas = z[k] 
    m.solve()
    x2mhe[k] = flow.model


## Cycle through measurement sequentially
for k in range(1, n_iter+1):
    print ('Cycle ' + str(k) + ' of ' + str(n_iter))
    time[k] = k

    # filtered bias update
    xb[k] = alpha * z[k] + (1.0-alpha) * xb[k-1] 


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(time,z,'kx',lw=2)
plt.plot(time,xb,'g--',lw=3)
plt.plot(time,x2mhe,'k-',lw=3)
plt.plot(time,x1mhe,'r.-',lw=3)
plt.plot(time,xtrue,'k:',lw=2)
plt.legend(['Measurement','Filtered Bias Update','Sq Error MHE','l_1-Norm MHE','Actual Value'])
plt.xlabel('Time (sec)')
plt.ylabel('Flow Rate (T/hr)')
plt.axis([0, time[n_iter], 32, 45])
plt.show()