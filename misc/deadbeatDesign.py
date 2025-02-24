import numpy as np
import scipy as sp
import sympy as sy
import matplotlib.pyplot as plt
import control as ct


#Simulation Constants
n = 1.107e-3
T = 0.5

umax = np.hstack([0.2, 0.2])

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


Crefy = np.array([0.,1.,0.,0.])
nr = Crefy.shape[0]
# Aaug = np.block([[Ad,np.zeros([nx,nr])],[Cref,np.zeros([nr,nr])]])
# Baug = np.block([[Bd],[np.zeros([nr,nu])]])
# Caug = np.hstack([Cref, np.zeros([nr,nr])])


# Cont = np.hstack([Baug, Aaug@Baug,Aaug**2@Baug,Aaug**3@Baug,Aaug**4@Baug])
# rankCont = np.linalg.matrix_rank(Cont)

# S = sp.linalg.solve_discrete_are(Aaug, Baug, Q, R)
# K = np.linalg.inv(R + np.transpose(Baug)@S@Baug)@(np.transpose(Baug)@S@Aaug)

# Kcon = ct.dlqr(Ad,Bd,Q,R, integral_action=Cref2)[0]
# Kp = Kcon[:,:nx]
# Ki = Kcon[:,nx:nx+nr]

Bd_prune = np.reshape(Bd[:,1],(nx,1))[[1,3],]
Ad_prune = Ad[[1,3],:][:,[1,3]]
C_prune = np.array([1,0])

A_aug = np.block([[Ad_prune,np.zeros([2,1])],[C_prune,np.eye(1)]])
B_aug = np.block([[Bd_prune],[np.zeros([1,1])]])
des_eig = np.array([0,0,0])
K_prune = ct.acker(A_aug,B_aug,des_eig)

act_eigs = np.linalg.eigvals(A_aug-B_aug@K_prune)

K_total = np.zeros([nu,nx])
K_total[1,1] = K_prune[0,0]
K_total[1,3] = K_prune[0,1]
K_i = np.vstack([0,K_prune[0,2]])


xtrue0 = np.array([20.,0.5,-0.2,0.])
xr = np.array([0.,1.,0.,0.])

# Simulate in closed loop
nsim = 100
xtrueP = np.empty([nx,nsim+1])
xintP = np.empty([1, nsim+1])
ctrls = np.empty([nu,nsim])
xtrueP[:,0] = xtrue0
xintP[0,0] = xtrue0[1] - xr[1]

for i in range(nsim):

    ctrl = -K_total@xtrueP[:,i] - K_i@xintP[:,i]
    # if (np.linalg.norm(ctrl) > umax[0]):
    #   ctrl[0] = ctrl[0]*(umax[0]/np.linalg.norm(ctrl))
    #   ctrl[1] = ctrl[1]*(umax[0]/np.linalg.norm(ctrl))
    ctrls[:,i] = ctrl
    xtrueP[:,i+1] = Ad@xtrueP[:,i] + Bd@ctrl
    xintP[:,i+1] = xintP[:,i] + (Crefy@xtrueP[:,i] - xr[1])


plt.figure(1)
x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)
# d1p = plt.subplot2grid((6, 3), (4, 0), rowspan=1, colspan=3)
# d2p = plt.subplot2grid((6, 3), (5, 0), rowspan=1, colspan=3)

xTime = [T*x for x in range(nsim+1)]
x1p.plot(xTime,xtrueP[0,:])
#x1p.plot(xTime,xestO[0,:])
x2p.plot(xTime,xtrueP[1,:])
#x2p.plot(xTime,xestO[1,:])
x3p.plot(xTime,xtrueP[2,:])
#x3p.plot(xTime,xestO[2,:])
x4p.plot(xTime,xtrueP[3,:])


plt.figure(2)

xTime = [T*x for x in range(nsim+1)]
plt.plot(xTime,xintP[0,:])



plt.figure(3)
u1p = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=3)
u2p = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=3)


uTime = [T*x for x in range(1,nsim+1)]
u1p.plot(uTime,ctrls[0,:nsim])
u2p.plot(uTime,ctrls[1,:nsim])
plt.show()