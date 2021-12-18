"""
Moving Horizon Estimation (MHE) uses dynamic optimization and a backward time horizon
of measurements to optimally adjust parameters and states. The data may include noise 
(random fluctuations), drift (gradual departure from true values), outliers 
(sudden and temporary departure from true values), or other inaccuracies. Nonlinear 
programming solvers are employed to numerically converge the dynamic optimization problem.

Exercise
Objective: Design an estimator to predict an unknown parameter and state variable. 
Use a model of the reactor and implement the estimator to detect the current states 
(temperature and concentration) as well as the unmeasured heat transfer coefficient (U).
A reactor is used to convert a hazardous chemical A to an acceptable chemical B in waste
stream before entering a nearby lake. This particular reactor is dynamically modeled as 
a Continuously Stirred Tank Reactor (CSTR) with a simplified kinetic mechanism that describes
the conversion of reactant A to product B with an irreversible and exothermic reaction. 
It is desired to maintain the temperature at a constant setpoint that maximizes the 
destruction of A (highest possible temperature). First, however, an estimator must predict
the concentration of A because there is no direct measurement of this quantity. 
The reaction kinetics and dynamic equations are well-known but there is a parameter 
in the model, the heat transfer coefficient UA, that is unknown.

Design an estimator to predict the concentration of A leaving the reactor and the heat 
transfer coefficient UA from the measured reactor temperature T and jacket temperature Tc. 
See a related CSTR case study for details on the model.
"""


from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Simulation
s = GEKKO(remote=False,name='cstr-sim')

# One step of simulation, discretization matches MHE
s.time = np.linspace(0,.1,2)

# Receive measurement from simulated control
Tc = s.MV(value=300,name='tc')
Tc.FSTATUS = 1 #receive measurement
Tc.STATUS = 0  #don't optimize

# Simulator variables
Ca = s.SV(value=.7, ub=1, lb=0,name='ca')
T = s.SV(value=335,lb=250,ub=500,name='t')

# Parameters
q = s.Param(value=100)
V = s.Param(value=100)
rho = s.Param(value=1000)
Cp = s.Param(value=0.239)
mdelH = s.Param(value=50000)
ER = s.Param(value=8750)
k0 = s.Param(value=7.2e10)
UA = s.Param(value=5e4)
Ca0 = s.Param(value=1)
T0 = s.Param(value=350)

# Variables
k = s.Var()
rate = s.Var()

# Rate equations
s.Equation(k==k0*s.exp(-ER/T))
s.Equation(rate==k*Ca)
# CSTR equations
s.Equation(V* Ca.dt() == q*(Ca0-Ca)-V*rate)
s.Equation(rho*Cp*V* T.dt() == q*rho*Cp*(T0-T) + V*mdelH*rate + UA*(Tc-T))

# Options
s.options.IMODE = 4 #dynamic simulation
s.options.NODES = 3
s.options.SOLVER = 3


# MHE
m = GEKKO(remote=False,name='cstr-mhe')

# 11 time points in horizon (0.1 each step)
m.time = np.linspace(0,1.0,21)

# Parameter to Estimate
UA_mhe = m.FV(value=1e4,name='ua')
UA_mhe.STATUS = 1  # estimate
UA_mhe.FSTATUS = 0 # no measurements
# Upper and lower bounds for optimizer
UA_mhe.LOWER = 30000
UA_mhe.UPPER = 100000

# Cooling Jacket Temperature
Tc_mhe = m.MV(value=300,name='tc')
Tc_mhe.STATUS = 0  # don't estimate
Tc_mhe.FSTATUS = 1 # receive measurement

# Reactor Temperature
T_mhe = m.CV(value=335,lb=250,ub=500,name='t')
T_mhe.FSTATUS = 1    # minimize error with measurement
T_mhe.MEAS_GAP = 0.1 # measurement deadband gap

# Reactor Concentration
Ca_mhe = m.SV(value=0.5, ub=1, lb=0,name='ca')

# Parameters
q = m.Param(value=100)
V = m.Param(value=100)
rho = m.Param(value=1000)
Cp = m.Param(value=0.239)
mdelH = m.Param(value=50000)
ER = m.Param(value=8750)
k0 = m.Param(value=7.2e10)
Ca0 = m.Param(value=1)
T0 = m.Param(value=350)

# Equation variables (2 other DOF from CV and FV)
k = m.Var()
rate = m.Var()

# Reaction equations
m.Equation(k==k0*m.exp(-ER/T_mhe))
m.Equation(rate==k*Ca_mhe)
# CSTR equations
m.Equation(V* Ca_mhe.dt() == q*(Ca0-Ca_mhe)-V*rate) # mol balance
m.Equation(rho*Cp*V* T_mhe.dt() == q*rho*Cp*(T0-T_mhe) \
           + V*mdelH*rate + UA_mhe*(Tc_mhe-T_mhe))  # energy balance

# Global Tuning
m.options.IMODE = 5  # MHE
m.options.EV_TYPE = 1
m.options.NODES = 3
m.options.SOLVER = 3 # IPOPT

# Cycles to run
cycles = 50

# step in the jacket cooling temperature at cycle 6
Tc_meas = np.empty(cycles)
Tc_meas[0:15] = 280
Tc_meas[5:cycles] = 300
dt = 0.1 # min
# time points for plot
time = np.linspace(0,cycles*dt-dt,cycles)

# allocate storage
Ca_meas = np.empty(cycles)
T_meas = np.empty(cycles)
UA_mhe_store = np.empty(cycles)
Ca_mhe_store = np.empty(cycles)
T_mhe_store = np.empty(cycles)

for i in range(cycles):
    # Process
    # input Tc (jacket cooling temperature)
    Tc.MEAS = Tc_meas[i]
    # simulate process model, 1 time step
    s.solve(disp=False)
    # retrieve Ca and T measurements
    Ca_meas[i] = Ca.MODEL
    T_meas[i] = T.MODEL

    # Estimator
    # input process measurements
    # input Tc (jacket cooling temperature)
    Tc_mhe.MEAS = Tc_meas[i]
    # input T (reactor temperature)
    T_mhe.MEAS = T_meas[i] #CV
    # Solve MHE
    m.solve(disp=False)
    # check if successful
    if m.options.APPSTATUS == 1:
        # retrieve solution
        UA_mhe_store[i] = UA_mhe.NEWVAL
        Ca_mhe_store[i] = Ca_mhe.MODEL
        T_mhe_store[i] = T_mhe.MODEL
    else:
        # failed solution
        UA_mhe_store[i] = 0
        Ca_mhe_store[i] = 0
        T_mhe_store[i] = 0

    print('MHE: Ca (est)=' + str(Ca_mhe_store[i]) + \
        ' Ca (actual)=' + str(Ca_meas[i]) + \
        ' UA (est)=' + str(UA_mhe_store[i]) + \
        ' UA (actual)=50000')

# plot results
plt.figure()
plt.subplot(411)
plt.plot(time,Tc_meas,'k-',lw=2)
plt.axis([0,time[-1],275,305])
plt.ylabel('Jacket T (K)')
plt.legend('T_c')

plt.subplot(412)
plt.plot([0,time[-1]],[50000,50000],'k--')
plt.plot(time,UA_mhe_store,'r:',lw=2)
plt.axis([0,time[-1],10000,100000])
plt.ylabel('UA')
plt.legend(['Actual UA','Predicted UA'],loc=4)

plt.subplot(413)
plt.plot(time,T_meas,'ro')
plt.plot(time,T_mhe_store,'b-',lw=2)
plt.axis([0,time[-1],300,340])
plt.ylabel('Reactor T (K)')
plt.legend(['Measured T','Predicted T'],loc=4)

plt.subplot(414)
plt.plot(time,Ca_meas,'go')
plt.plot(time,Ca_mhe_store,'m-',lw=2)
plt.axis([0,time[-1],.6,1])
plt.ylabel('Reactor C_a (mol/L)')
plt.legend(['Measured C_a','Predicted C_a'],loc=4)
plt.show()