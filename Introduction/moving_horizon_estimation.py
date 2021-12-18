from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt  

# Estimator Model
m = GEKKO()
m.time = p.time
# Parameters
m.u = m.MV(value=u_meas) #input
m.K = m.FV(value=1, lb=1, ub=3)    # gain
m.tau = m.FV(value=5, lb=1, ub=10) # time constant
# Variables
m.x = m.SV() #state variable
m.y = m.CV(value=y_meas) #measurement
# Equations
m.Equations([m.tau * m.x.dt() == -m.x + m.u,
             m.y == m.K * m.x])
# Options
m.options.IMODE = 5 #MHE
m.options.EV_TYPE = 1
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
m.K.DMAX = 2.0
m.tau.DMAX = 4.0
# MEAS_GAP = dead-band for measurement / model mismatch
m.y.MEAS_GAP = 0.25

# solve
m.solve(disp=False)

# Plot results
plt.subplot(2,1,1)
plt.plot(m.time,u_meas,'b:',label='Input (u) meas')
plt.legend()
plt.subplot(2,1,2)
plt.plot(m.time,y_meas,'gx',label='Output (y) meas')
plt.plot(p.time,p.y.value,'k-',label='Output (y) actual')
plt.plot(m.time,m.y.value,'r--',label='Output (y) estimated')
plt.legend()
plt.show()