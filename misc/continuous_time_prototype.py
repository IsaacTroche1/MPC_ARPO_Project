import numpy as np
import scipy as sp
import sympy as sy
from scipy import sparse
import matplotlib.pyplot as plt


n = 1.107e-3
T = 0.5
Tcont = 0.001
time_final = 20
nsimD = int(time_final/T)
nsimC = int(time_final/Tcont)

xTimeD = np.arange(0, time_final, T)
xTimeC = np.arange(0, time_final, Tcont)

#Using step input
ramp_slope = 0.2
ctrlD = np.empty([2,nsimD])
ctrlD[0,:] = np.arange(0,nsimD*ramp_slope,ramp_slope)
ctrlD[1,:] = np.arange(0,nsimD*ramp_slope,ramp_slope)

ctrlC = np.empty([2,nsimC])
j = 0
for i in range(nsimC):
    if ((j<nsimD) and (xTimeC[i] == xTimeD[j])):
        ctrlC[:,i] = ctrlD[:,j]
        j = j + 1
    else:
        ctrlC[:,i] = ctrlC[:,i-1]




#Initial conditions
x0 = np.array([100.,10.,0.,0.])

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

#Discretize
Ad = sparse.csc_matrix(sp.linalg.expm(Ap*T))
x = sy.symbols('x')
Ax = sy.Matrix(Ap)*x
eAx = Ax.exp()
eAxInt = np.empty([Ap.shape[0],Ap.shape[0]])
for (i,j), func_ij in np.ndenumerate(eAx):
    func = sy.lambdify((x),func_ij)
    eAxInt[i,j] = sp.integrate.quad(func,0.,T)[0]
Bd = sparse.csc_matrix(eAxInt@Bp)

#Discrete time simulation
x_valsD = np.empty([4, nsimD])
x_valsD[:,0] = x0
time = 0
for i in range(nsimD-1):
    x_valsD[:,i+1] = Ad@x_valsD[:,i] + Bd@ctrlD[:,i]


x_valsC = np.empty([4, nsimC])
x_valsC[:,0] = x0
time = 0
for i in range(nsimC-1):
    soln = sp.integrate.solve_ivp(stateEqn, (time, time + Tcont), x_valsC[:, i], args=(ctrlC[:, i],))
    x_valsC[:,i+1] = soln.y[:,-1]
    time = time + Tcont

# x_valsC_rk = np.empty([4, nsimC])
# x_valsC_rk[:,0] = x0
# x_valsC_fe = np.empty([4, nsimC])
# x_valsC_fe[:,0] = x0
# ctrlR = np.array([[1],[1]])
# time = 0
# for i in range(nsimC-1):
#     x_valsC_fe[:,i+1] = x_valsC_fe[:,i] + (Ap@x_valsC_fe[:,i] + Bp@ctrlR[:,0])*Tcont
#     soln = sp.integrate.solve_ivp(stateEqn, (time,time+Tcont), x_valsC_rk[:,i], args = (ctrlR[:,0],))
#     x_valsC_rk[:,i+1] = soln.y[:,-1]
#     time = time + Tcont


# plt.figure(1)
# x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
# x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
# x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
# x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)
#
# x1p.plot(xTimeC, x_valsC_fe[0,:])
# x1p.plot(xTimeC, x_valsC_rk[0,:])
# x1p.title.set_text('States')
# x1p.legend(['Forward Euler', 'RK4'])
#
# x2p.plot(xTimeC, x_valsC_fe[1,:])
# x2p.plot(xTimeC, x_valsC_rk[1,:])
#
# x3p.plot(xTimeC, x_valsC_fe[2,:])
# x3p.plot(xTimeC, x_valsC_rk[2,:])
#
# x4p.plot(xTimeC, x_valsC_fe[3,:])
# x4p.plot(xTimeC, x_valsC_rk[3,:])
# plt.show()



plt.figure(1)
x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)

x1p.plot(xTimeD, x_valsD[0,:])
x1p.plot(xTimeC, x_valsC[0,:],'--')
x1p.title.set_text('States')
x1p.legend(['Discrete', 'Continuous'])

x2p.plot(xTimeD, x_valsD[1,:])
x2p.plot(xTimeC, x_valsC[1,:],'--')

x3p.plot(xTimeD, x_valsD[2,:])
x3p.plot(xTimeC, x_valsC[2,:],'--')

x4p.plot(xTimeD, x_valsD[3,:])
x4p.plot(xTimeC, x_valsC[3,:],'--')

x1p.title.set_text('Discrete and Continuous Time States')
x1p.set_ylabel('$\mathregular{\delta}$x (m)')
x1p.xaxis.set_visible(False)
x2p.set_ylabel('$\mathregular{\delta}$y (m)')
x2p.xaxis.set_visible(False)
x3p.set_ylabel('$\mathregular{\delta\dot{x}}$ (m/s)')
x3p.xaxis.set_visible(False)
x4p.set_ylabel('$\mathregular{\delta\dot{y}}$ (m/s)')
x4p.set_xlabel('Time (s)')

plt.figure(2)
u1p = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=3)
u2p = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=3)

u1p.plot(xTimeD, ctrlD[0,:])
u1p.plot(xTimeC, ctrlC[0,:])
u1p.title.set_text('Inputs')
u1p.legend(['Discrete','Continuous'])
u2p.plot(xTimeD, ctrlD[1,:])
u2p.plot(xTimeC, ctrlC[1,:])

u1p.title.set_text('Discrete and Continuous Commands')
u1p.set_ylabel('$\mathregular{u_x}$ $\mathregular{(m/s^2)}$')
u2p.set_ylabel('$\mathregular{u_y}$ $\mathregular{(m/s^2)}$')
u2p.set_xlabel('Time (s)')

plt.show()



