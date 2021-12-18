'''Many Single Input, Single Output (SISO) dynamic systems can be represented empirically
with a First Order Plus Dead Time (FOPDT) model. Model identification involves fitting 
parameters in a dynamic continuous or discrete form of the FOPDT model. The unknown 
parameters for this system include the time constant (τ), gain (K), and sometimes 
dead-time (θ). τ(dx/dt) = -x + Ku(t - θ)'''

from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data and parse into columns
url = 'http://apmonitor.com/do/uploads/Main/tclab_siso_data.txt'
data = pd.read_csv(url)
t = data['time']
u = data['voltage']
y = data['temperature']

# generate time-series model
m = GEKKO()

# system identification
na = 2 # output coefficients
nb = 2 # input coefficients
yp,p,K = m.sysid(t,u,y,na,nb,pred='meas')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,u)
plt.legend([r'$V_1$ (mV)'])
plt.ylabel('MV Voltage (mV)')
plt.subplot(2,1,2)
plt.plot(t,y)
plt.plot(t,yp)
plt.legend([r'$T_{1meas}$',r'$T_{1pred}$'])
plt.ylabel('CV Temp (degF)')
plt.xlabel('Time')
plt.savefig('sysid.png')

# step test model
yc,uc = m.arx(p)

# steady state initialization
m.options.IMODE = 1
m.solve(disp=False)

# dynamic simulation (step test)
m.time = np.linspace(0,240,241)
m.options.TIME_SHIFT=0
m.options.IMODE = 4
m.solve(disp=False)

# step for first MV (Heater 1)
uc[0].value = np.zeros(len(m.time))
uc[0].value[5:] = 100
m.solve(disp=False)

plt.figure()
plt.subplot(2,1,1)
plt.title('Step Test 1')
plt.plot(m.time,uc[0].value,'b-',label=r'$H_1$')
plt.ylabel('Heater (V)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(m.time,yc[0].value,'r--',label=r'$T_1$')
plt.ylabel('Temperature (degF)')
plt.legend()
plt.xlabel('Time (sec)')
plt.legend()
plt.savefig('step_test.png')
plt.show()