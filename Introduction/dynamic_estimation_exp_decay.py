"""
Estimate the parameter k in the exponential decay equation:
dx/dt = -kx by minimizing the error between the predicted and measured 
x values. The x values are measured at the following time intervals.

Use an initial condition of x=2 that matches the data. Verify the solution of 
x with the analytic expression x(t)=2exp(-kt).
"""

from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data set 1
t_data1 = [0.0,  0.1,  0.2, 0.4, 0.8, 1.00]
x_data1 = [2.0,  1.6,  1.2, 0.7, 0.3, 0.15]

# data set 2
t_data2 = [0.0,  0.15, 0.25, 0.45, 0.85, 0.95]
x_data2 = [3.6,  2.25, 1.75, 1.00, 0.35, 0.20]

# combine with dataframe join
data1 = pd.DataFrame({'Time':t_data1,'x1':x_data1})
data2 = pd.DataFrame({'Time':t_data2,'x2':x_data2})
data1.set_index('Time', inplace=True)
data2.set_index('Time', inplace=True)
data = data1.join(data2,how='outer')
print(data.head())

# indicate which points are measured
z1 = (data['x1']==data['x1']).astype(int) # 0 if NaN
z2 = (data['x2']==data['x2']).astype(int) # 1 if number

# replace NaN with any number (0)
data.fillna(0,inplace=True)

m = GEKKO(remote=False)

# measurements
xm = m.Array(m.Param,2)
xm[0].value = data['x1'].values
xm[1].value = data['x2'].values

# index for objective (0=not measured, 1=measured)
zm = m.Array(m.Param,2)
zm[0].value=z1
zm[1].value=z2

m.time = data.index
x = m.Array(m.Var,2)                   # fit to measurement
x[0].value=x_data1[0]; x[1].value=x_data2[0]

k = m.FV(); k.STATUS = 1               # adjustable parameter
for i in range(2):
    m.free_initial(x[i])               # calculate initial condition
    m.Equation(x[i].dt()== -k * x[i])  # differential equations
    m.Minimize(zm[i]*(x[i]-xm[i])**2)  # objectives

m.options.IMODE = 5   # dynamic estimation
m.options.NODES = 2   # collocation nodes
m.solve(disp=True)    # solve
k = k.value[0]
print('k = '+str(k))

# plot solution
plt.plot(m.time,x[0].value,'b.--',label='Predicted 1')
plt.plot(m.time,x[1].value,'r.--',label='Predicted 2')
plt.plot(t_data1,x_data1,'bx',label='Measured 1')
plt.plot(t_data2,x_data2,'rx',label='Measured 2')
plt.legend(); plt.xlabel('Time'); plt.ylabel('Value')
plt.xlabel('Time');
plt.show()
