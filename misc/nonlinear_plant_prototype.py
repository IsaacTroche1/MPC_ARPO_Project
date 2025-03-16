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


ctrlC = np.ones([2,nsimC])

#Initial conditions
x0 = np.array([0.,0.,0.,0.])

Ap = np.array([
    [0.,      0.,     1., 0.],
    [0.,      0.,     0., 1.],
    [3*n**2,      0.,     0., 2*n],
    [0.,  0.,     -2*n, 0.],
    ])

Bp = np.array([
    [0.,      0.],
    [0.,  0.],
    [1.,  0.],
    [0.,     1.],
    ])


def stateEqn(t, x, u):
    Ap = np.array([
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [3 * n ** 2, 0., 0., 2 * n],
        [0., 0., -2 * n, 0.],
    ])

    Bp = np.array([
        [0., 0.],
        [0., 0.],
        [1., 0.],
        [0., 1.],
    ])

    # u = np.array([[1],[1]])
    dxdt = Ap@x + Bp@u
    return dxdt

def stateEqnN(t, x, u):

    dxdt = [None] * 4

    dxdt[0] = x[2]
    dxdt[1] = x[3]
    dxdt[2] = 2*n*x[3] + (n**2)*x[0] - (mu*(R_T + x[0]))/(((R_T + x[0])**2+x[1]**2)**(3/2)) + mu/(R_T**2) + u[0]
    dxdt[3] = -2*n*x[2] + (n**2)*x[1] - (mu*x[1])/(((R_T + x[0])**2+x[1]**2)**(3/2)) + u[1]

    return dxdt


# ctrlC = np.zeros([2,nsimC])

x_valsC_L = np.empty([4, nsimC])
x_valsC_N = np.empty([4, nsimC])
x_valsC_L[:,0] = x0
x_valsC_N[:,0] = x0
time = 0
for i in range(nsimC-1):
    solnL = sp.integrate.solve_ivp(stateEqn, (time, time + Tcont), x_valsC_L[:, i], args=(ctrlC[:, i],))
    solnN = sp.integrate.solve_ivp(stateEqnN, (time, time + Tcont), x_valsC_N[:, i], args=(ctrlC[:, i],))
    x_valsC_L[:,i+1] = solnL.y[:,-1]
    x_valsC_N[:,i+1] = solnN.y[:,-1]

    time = time + Tcont




plt.figure(1)
x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)

x1p.plot(xTimeC, x_valsC_L[0,:])
x1p.plot(xTimeC, x_valsC_N[0,:],'--')
x1p.title.set_text('States')
x1p.legend(['Linear Plant', 'Nonlinear Plant'])

x2p.plot(xTimeC, x_valsC_L[1,:])
x2p.plot(xTimeC, x_valsC_N[1,:],'--')

x3p.plot(xTimeC, x_valsC_L[2,:])
x3p.plot(xTimeC, x_valsC_N[2,:],'--')

x4p.plot(xTimeC, x_valsC_L[3,:])
x4p.plot(xTimeC, x_valsC_N[3,:],'--')

x1p.title.set_text('Linear/Nonlinear Step Response')
x1p.set_ylabel('$\mathregular{\delta}$x (m)')
x1p.xaxis.set_visible(False)
x2p.set_ylabel('$\mathregular{\delta}$y (m)')
x2p.xaxis.set_visible(False)
x3p.set_ylabel('$\mathregular{\delta\dot{x}}$ (m/s)')
x3p.xaxis.set_visible(False)
x4p.set_ylabel('$\mathregular{\delta\dot{y}}$ (m/s)')
x4p.set_xlabel('Time (s)')


plt.show()
