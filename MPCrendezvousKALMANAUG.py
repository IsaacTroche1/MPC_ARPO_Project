import osqp
import numpy as np
import scipy as sp
import sympy as sy
import math
from scipy import sparse
from numpy import random
import matplotlib.pyplot as plt

#Noise characteristics
sig_x = 15
sig_y = 15
sig_xd = 0
sig_yd = 0
sigMat = np.diag([sig_x, sig_y, sig_xd, sig_yd])
Qw = np.diag([sig_x**2, sig_y**2, sig_xd**2, sig_yd**2])
random.seed(123)

#Simulation Constants
n = 1.107e-3
T = 0.5

#Dynamics
Ap = np.array([
  [0.,      0.,     1., 0.],
  [0.,      0.,     0., 1.],
  [3*n**2,      0.,     0., 2*n],
  [0.,  0.,     -2*n, 0.],
])

Adi = np.eye(2)
ndi = Adi.shape[0]

Bp = np.array([
  [0.,      0.],
  [0.,  0.],
  [1.,  0.],
  [0.,     1.],
])

Bdi = 1*np.eye(2) #try with T instead of 1

Cm = np.array([
  [1., 0., 0., 0.],
  [0., 1., 0., 0.],
])

Cdi = np.eye(2)

[nx, nu] = Bp.shape
nym = Cm.shape[0]

#Discretize
Ad = sp.linalg.expm(Ap*T)
x = sy.symbols('x')
Ax = sy.Matrix(Ap)*x
eAx = Ax.exp()
eAxInt = np.empty([Ap.shape[0],Ap.shape[0]])
for (i,j), func_ij in np.ndenumerate(eAx):
    func = sy.lambdify((x),func_ij)
    eAxInt[i,j] = sp.integrate.quad(func,0.,T)[0]
Bd = eAxInt@Bp

Q = 1e+02*np.diag([10., 10., 1, 1])
R = 10**2*np.eye(2)

#LQR Stuff
S = sp.linalg.solve_discrete_are(Ad, Bd, Q, R)
K = np.asarray(np.linalg.inv(R + np.transpose(Bd)@S@Bd)@(np.transpose(Bd)@S@Ad))

#Construct augmented state observer system
Ao = sp.linalg.block_diag(Ad, Adi, np.zeros([2,2]))
Bo = sp.linalg.block_diag(Bd, Bdi, np.zeros([2,2]))
Co = np.hstack([Cm, Cdi, np.zeros([2,2])])

Ao = sp.linalg.block_diag(Ad, Adi)
Bo = sp.linalg.block_diag(Bd, Bdi)
Co = np.hstack([Cm, Cdi])
Co = np.hstack([Cm, np.zeros([2,2])])

Ao[0,4] = 1.
Ao[1,5] = 1.

#Test for observability of modes
eigs = np.linalg.eigvals(Ao)
pbh0 = np.vstack([eigs[0]*np.eye(nx+ndi)-Ao, Co])
rank0 = np.linalg.matrix_rank(pbh0)
pbh1 = np.vstack([eigs[1]*np.eye(nx+ndi)-Ao, Co])
rank1 = np.linalg.matrix_rank(pbh1)
pbh2 = np.vstack([eigs[2]*np.eye(nx+ndi)-Ao, Co])
rank2 = np.linalg.matrix_rank(pbh2)
pbh3 = np.vstack([eigs[3]*np.eye(nx+ndi)-Ao, Co])
rank3 = np.linalg.matrix_rank(pbh3)
pbh4 = np.vstack([eigs[4]*np.eye(nx+ndi)-Ao, Co])
rank4 = np.linalg.matrix_rank(pbh4)
pbh5 = np.vstack([eigs[5]*np.eye(nx+ndi)-Ao, Co])
rank5 = np.linalg.matrix_rank(pbh5)


xtrue0 = np.array([100.,67.,10.,-3.])
xest0 = np.array([100.,67.,10.,-3., 0., 0.]) #note this assumes we dont know the inital distrubance value (zeros)
Pest = 10*np.eye(nx+ndi)

# Simulate in closed loop
nsim = 100
xtrueP = np.empty([nx,nsim+1])
xestO = np.empty([nx+ndi,nsim+1])
noiseStored = np.empty([nx,nsim+1])
xtrueP[:,0] = xtrue0
xestO[:,0] = xest0  
noiseVec = sigMat@random.normal(0, 1, 4)
noiseStored[:,0] = noiseVec
#xestO[:,0] = np.array([100.,67.,10.,-3., noiseVec[0], noiseVec[1]])

Bou = np.vstack([Bd, np.zeros([2,2])])
Bnoise = np.vstack([np.zeros([nx,ndi]), np.eye(ndi)])
Qw = np.diag([sig_x**2, sig_y**2])
Qw = Bnoise@Qw@np.transpose(Bnoise)
for i in range(nsim):

    ctrl = -K@xtrueP[:,i]
    xtrueP[:,i+1] = Ad@xtrueP[:,i] + Bd@ctrl + noiseVec

    #Measurement and state estimate
    xnom = Ao@xestO[:,i] + Bou@ctrl
    Pest = Ao@Pest@np.transpose(Ao) + Qw
    L = Pest@np.transpose(Co)@sp.linalg.inv(Co@Pest@np.transpose(Co))
    ymeas = Cm@xtrueP[:,i]
    xestO[:,i+1] = xnom + L@(ymeas - Co@xnom)
    Pest = (np.eye(nx+ndi) - L@Co)@Pest

    #Inject noise into plant
    noiseVec = sigMat@random.normal(0, 1, 4)
    noiseStored[:,i+1] = noiseVec

plt.figure(2)
x1p = plt.subplot2grid((6, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((6, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((6, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((6, 3), (3, 0), rowspan=1, colspan=3)
d1p = plt.subplot2grid((6, 3), (4, 0), rowspan=1, colspan=3)
d2p = plt.subplot2grid((6, 3), (5, 0), rowspan=1, colspan=3)

xTime = [T*x for x in range(nsim+1)]
x1p.plot(xTime,xtrueP[0,:])
x1p.plot(xTime,xestO[0,:])
x2p.plot(xTime,xtrueP[1,:])
x2p.plot(xTime,xestO[1,:])
x3p.plot(xTime,xtrueP[2,:])
x3p.plot(xTime,xestO[2,:])
x4p.plot(xTime,xtrueP[3,:])
x4p.plot(xTime,xestO[3,:])
d1p.plot(xTime,noiseStored[0,:])
d1p.plot(xTime,xestO[4,:])
d2p.plot(xTime,noiseStored[1,:])
d2p.plot(xTime,xestO[5,:])
plt.show()