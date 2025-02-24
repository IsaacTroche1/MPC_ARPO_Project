import osqp
import numpy as np
import scipy as sp
import sympy as sy
import math
from scipy import sparse
from numpy import random
import matplotlib.pyplot as plt

#Noise characteristics
sig_x = 10
sig_y = 10
sig_xd = 0
sig_yd = 0
sigMat = np.diag([sig_x, sig_y, sig_xd, sig_yd])
Qw = np.diag([sig_x**2, sig_y**2, sig_xd**2, sig_yd**2])
random.seed(123)

#Simulation Constants
n = 1.107e-3
T = 0.5

# Discrete time model of a quadcopter
A = np.array([
  [0.,      0.,     1., 0.],
  [0.,      0.,     0., 1.],
  [3*n**2,      0.,     0., 2*n],
  [0.,  0.,     -2*n, 0.],
])
B = np.array([
  [0.,      0.],
  [0.,  0.],
  [1.,  0.],
  [0.,     1.],
])
Cm = np.array([
  [1., 0., 0., 0.],
  [0., 1., 0., 0.],
])
[nx, nu] = B.shape
nym = Cm.shape[0]

#Discretize
Ad = sp.linalg.expm(A*T)
x = sy.symbols('x')
Ax = sy.Matrix(A)*x
eAx = Ax.exp()
eAxInt = np.empty([A.shape[0],A.shape[0]])
for (i,j), func_ij in np.ndenumerate(eAx):
    func = sy.lambdify((x),func_ij)
    eAxInt[i,j] = sp.integrate.quad(func,0.,T)[0]
Bd = eAxInt@B

Q = 1e+02*np.diag([10., 10., 1, 1])
R = 10**2*np.eye(2)

#LQR Stuff
S = sp.linalg.solve_discrete_are(Ad, Bd, Q, R)
K = np.asarray(np.linalg.inv(R + np.transpose(Bd)@S@Bd)@(np.transpose(Bd)@S@Ad))

#Solve Riccati for Kalman gain
P = sp.linalg.solve_discrete_are(Ad, np.transpose(Cm), Qw, np.zeros([nym,nym]))
L = (Ad@P@np.transpose(Cm))@np.linalg.inv(Cm@P@np.transpose(Cm))

xtrue0 = np.array([100.,67.,10.,-3.])
xest0 = np.array([100.,67.,10.,-3.])

# Simulate in closed loop
nsim = 1000
xtrue = np.empty([nx,nsim+1])
xest = np.empty([nx,nsim+1])
noiseStored = np.empty([nx,nsim+1])
xtrue[:,0] = xtrue0
xest[:,0] = xest0
noiseVec = sigMat@random.rand(4)
noiseStored[:,0] = noiseVec
for i in range(nsim):

    ctrl = -K@xtrue[:,i]
    xtrue[:,i+1] = Ad@xtrue[:,i] + Bd@ctrl + noiseVec

    #Measurement and state estimate
    ymeas = Cm@xtrue[:,i]
    xnom = Ad@xest[:,i] + Bd@ctrl
    xest[:,i+1] = xnom + L@(ymeas - Cm@xnom)

    #Inject noise into plant
    noiseVec = sigMat@random.normal(0, 1, 4)
    noiseStored[:,i+1] = noiseVec

plt.figure(2)
x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3, sharey = x3p)

xTime = [T*x for x in range(nsim+1)]
x1p.plot(xTime,xtrue[0,:])
x1p.plot(xTime,xest[0,:])
x2p.plot(xTime,xtrue[1,:])
x2p.plot(xTime,xest[1,:])
x3p.plot(xTime,xtrue[2,:])
x3p.plot(xTime,xest[2,:])
x4p.plot(xTime,xtrue[3,:])
x4p.plot(xTime,xest[3,:])
plt.show()