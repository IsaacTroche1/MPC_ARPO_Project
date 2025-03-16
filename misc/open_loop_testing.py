import numpy as np
import scipy as sp
import sympy as sy
from scipy import sparse
import matplotlib.pyplot as plt


n = 1.107e-3
h = 500e+03
re = 6378.1e+03
R_T = h + re
mu = (n**2)*(R_T**3)



T = 0.5
Tcont = 0.001
time_final = 20
nsimC = int(time_final/Tcont)


xTimeC = np.arange(0, time_final, Tcont)


ctrlCxy = np.ones([2,nsimC])
ctrlCx = np.vstack([np.ones(nsimC),np.zeros(nsimC)])
ctrlCy = np.vstack([np.zeros(nsimC),np.ones(nsimC)])

ctrlCxy = np.zeros([2,nsimC])


#Initial conditions
x0 = np.array([0.,0.,0.,0.])


def stateEqnN(t, x, u):

    dxdt = [None] * 4

    dxdt[0] = x[2]
    dxdt[1] = x[3]
    dxdt[2] = 2*n*x[3] + (n**2)*x[0] - (mu*(R_T + x[0]))/(((R_T + x[0])**2+x[1]**2)**(3/2)) + mu/(R_T**2) + u[0]
    dxdt[3] = -2*n*x[2] + (n**2)*x[1] - (mu*x[1])/(((R_T + x[0])**2+x[1]**2)**(3/2)) + u[1]

    return dxdt


# ctrlC = np.zeros([2,nsimC])

x_valsC_Nxy = np.empty([4, nsimC])
x_valsC_Nx = np.empty([4, nsimC])
x_valsC_Ny = np.empty([4, nsimC])
x_valsC_Nxy[:,0] = x0
x_valsC_Nx[:,0] = x0
x_valsC_Ny[:,0] = x0
time = 0
for i in range(nsimC-1):
    solnNxy = sp.integrate.solve_ivp(stateEqnN, (time, time + Tcont), x_valsC_Nxy[:, i], args=(ctrlCxy[:, i],))
    x_valsC_Nxy[:,i+1] = solnNxy.y[:,-1]
    solnNx = sp.integrate.solve_ivp(stateEqnN, (time, time + Tcont), x_valsC_Nx[:, i], args=(ctrlCx[:, i],))
    x_valsC_Nx[:, i + 1] = solnNx.y[:, -1]
    solnNy = sp.integrate.solve_ivp(stateEqnN, (time, time + Tcont), x_valsC_Ny[:, i], args=(ctrlCy[:, i],))
    x_valsC_Ny[:, i + 1] = solnNy.y[:, -1]

    time = time + Tcont




plot = plt.figure(1)
x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)


x1p.plot(xTimeC, x_valsC_Ny[0,:])

x2p.plot(xTimeC, x_valsC_Ny[1,:])

x3p.plot(xTimeC, x_valsC_Ny[2,:])

x4p.plot(xTimeC, x_valsC_Ny[3,:])

x1p.title.set_text('Open-loop $\mathregular{u_y}$ Step Response')
x1p.set_ylabel('$\mathregular{\delta}$x (m)')
x1p.xaxis.set_visible(False)
x2p.set_ylabel('$\mathregular{\delta}$y (m)')
x2p.xaxis.set_visible(False)
x3p.set_ylabel('$\mathregular{\delta\dot{x}}$ (m/s)')
x3p.xaxis.set_visible(False)
x4p.set_ylabel('$\mathregular{\delta\dot{y}}$ (m/s)')
x4p.set_xlabel('Time (s)')

plot.align_labels()


plot = plt.figure(2)
x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)


x1p.plot(xTimeC, x_valsC_Nx[0,:])

x2p.plot(xTimeC, x_valsC_Nx[1,:])

x3p.plot(xTimeC, x_valsC_Nx[2,:])

x4p.plot(xTimeC, x_valsC_Nx[3,:])

x1p.title.set_text('Open-loop $\mathregular{u_x}$ Step Response')
x1p.set_ylabel('$\mathregular{\delta}$x (m)')
x1p.xaxis.set_visible(False)
x2p.set_ylabel('$\mathregular{\delta}$y (m)')
x2p.xaxis.set_visible(False)
x3p.set_ylabel('$\mathregular{\delta\dot{x}}$ (m/s)')
x3p.xaxis.set_visible(False)
x4p.set_ylabel('$\mathregular{\delta\dot{y}}$ (m/s)')
x4p.set_xlabel('Time (s)')

plot.align_labels()

plot = plt.figure(3)
x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)


x1p.plot(xTimeC, x_valsC_Nxy[0,:])

x2p.plot(xTimeC, x_valsC_Nxy[1,:])

x3p.plot(xTimeC, x_valsC_Nxy[2,:])

x4p.plot(xTimeC, x_valsC_Nxy[3,:])

x1p.title.set_text('Open-loop Zero-input Response')
x1p.set_ylabel('$\mathregular{\delta}$x (m)')
x1p.xaxis.set_visible(False)
x2p.set_ylabel('$\mathregular{\delta}$y (m)')
x2p.xaxis.set_visible(False)
x3p.set_ylabel('$\mathregular{\delta\dot{x}}$ (m/s)')
x3p.xaxis.set_visible(False)
x4p.set_ylabel('$\mathregular{\delta\dot{y}}$ (m/s)')
x4p.set_xlabel('Time (s)')

plot.align_labels()

plt.show()