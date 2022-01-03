"""
Objective: Implement MHE with an without Model Predictive Control (MPC). 
Describe the difference between tracking performance (agreement between model 
and measured values, MHE) and predictive performance (model parameters that 
predict into the future, MPC). Tune the estimator to improve control performance. 
Show how overly aggressive tracking(estimating) performance may degrade the control 
performance even though the estimator performance (difference between measured 
and modeled output) is acceptable. An overly aggressive estimator may give 
different parameter values (K and tau) for each cycle.

A linear, first order process is described by a gain K and time constant τ. 
This model is expressed in differential form as:

τ(dy/dt)= -y + Ku For this exercise, change the process model gain and time 
constant to unique values between 1 and 10. For example, the gain is 5.0 and 
the time constant is 10.0 in the Simulink model below in Laplace form as 
(5/10s+1). There are three models in this exercise including the process, 
the estimator, and the controller. The first part of the exercise is to tune 
just the estimator(MHE without MPC). While an estimator can be well tuned to 
track measured values, it may be undesirable to have large parameter variations
that are passed with each cycle to a model predictive controller.

Observe the estimator performance as new measurements arrive. Is the estimator 
able to reconstruct the unmeasured model parameters from the input and output data?
"""

#Import packages
import numpy as np
from random import random
from gekko import GEKKO
import matplotlib.pyplot as plt

# Process
p = GEKKO()

p.time = [0,.5]

#Parameters
p.u = p.MV()
p.K = p.Param(value=1) #gain
p.tau = p.Param(value=5) #time constant

#variable
p.y = p.SV() #measurement

#Equations
p.Equation(p.tau * p.y.dt() == -p.y + p.K * p.u)

#options
p.options.IMODE = 4

# MHE Model
m = GEKKO()

m.time = np.linspace(0,20,41) #0-20 by 0.5 -- discretization must match simulation

#Parameters
m.u = m.MV() #input
m.K = m.FV(value=3, lb=1, ub=3) #gain
m.tau = m.FV(value=4, lb=1, ub=10) #time constant

#Variables
m.y = m.CV() #measurement

#Equations
m.Equation(m.tau * m.y.dt() == -m.y + m.K*m.u)

#Options
m.options.IMODE = 5 #MHE
m.options.EV_TYPE = 1
m.options.DIAGLEVEL = 0

# STATUS = 0, optimizer doesn't adjust value
# STATUS = 1, optimizer can adjust
m.u.STATUS = 0
m.K.STATUS = 1
m.tau.STATUS = 1
m.y.STATUS = 1

# FSTATUS = 0, no measurement
# FSTATUS = 1, measurement used to update model
m.u.FSTATUS = 1
m.K.FSTATUS = 0
m.tau.FSTATUS = 0
m.y.FSTATUS = 1

# DMAX = maximum movement each cycle
m.K.DMAX = 1
m.tau.DMAX = .1

# MEAS_GAP = dead-band for measurement / model mismatch
m.y.MEAS_GAP = 0.25

m.y.TR_INIT = 1

#problem configuration
# number of cycles
cycles = 50
# noise level
noise = 0.25

#run process, estimator and control for cycles
y_meas = np.empty(cycles)
y_est = np.empty(cycles)
k_est = np.empty(cycles)
tau_est = np.empty(cycles)
u_cont = np.empty(cycles)
u = 2.0

# Create plot
plt.figure(figsize=(10,7))
plt.ion()
plt.show()

for i in range(cycles):
    # change input (u)
    if i==10:
        u = 3.0
    elif i==20:
        u = 4.0
    elif i==30:
        u = 1.0
    elif i==40:
        u = 3.0
    u_cont[i] = u

    ## process simulator
    #load u value
    p.u.MEAS = u_cont[i]
    #simulate
    p.solve()
    #load output with white noise
    y_meas[i] = p.y.MODEL + (random()-0.5)*noise

    ## estimator
    #load input and measured output
    m.u.MEAS = u_cont[i]
    m.y.MEAS = y_meas[i]
    #optimize parameters
    m.solve()
    #store results
    y_est[i] = m.y.MODEL
    k_est[i] = m.K.NEWVAL
    tau_est[i] = m.tau.NEWVAL

    plt.clf()
    plt.subplot(4,1,1)
    plt.plot(y_meas[0:i])
    plt.plot(y_est[0:i])
    plt.legend(('meas','pred'))
    plt.ylabel('y')
    plt.subplot(4,1,2)
    plt.plot(np.ones(i)*p.K.value[0])
    plt.plot(k_est[0:i])
    plt.legend(('actual','pred'))
    plt.ylabel('k')
    plt.subplot(4,1,3)
    plt.plot(np.ones(i)*p.tau.value[0])
    plt.plot(tau_est[0:i])
    plt.legend(('actual','pred'))
    plt.ylabel('tau')
    plt.subplot(4,1,4)
    plt.plot(u_cont[0:i])
    plt.legend('u')
    plt.draw()
    plt.pause(0.05)
