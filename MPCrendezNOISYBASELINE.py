import osqp
import numpy as np
import scipy as sp
import sympy as sy
import math
from scipy import sparse
from numpy import random
import matplotlib.pyplot as plt

#Noise characteristics
sig_x = 0.0
sig_y = 0.0
sig_xd = 0
sig_yd = 0
sigMat = np.diag([sig_x, sig_y, sig_xd, sig_yd])
Qw = np.diag([sig_x**2, sig_y**2, sig_xd**2, sig_yd**2])
random.seed(123)

#Simulation Constants
gam = 10*(np.pi/180)
rp = 2.5
rtot = 2.49
phi = 0*(np.pi/180)
n = 1.107e-3
T = 0.5

#for debugging
def sparseBlockIndex(arr, blkinxs, nrows, ncols = np.nan):
    arr = arr.toarray()
    blx = blkinxs[0]
    bly = blkinxs[1]
    if ~np.isnan(ncols):
        rowIndices = np.squeeze(blx*nrows*np.ones((1,nrows)) + range(nrows))
        colIndices = np.squeeze(bly*ncols*np.ones((1,ncols)) + range(ncols))
    else:
        rowIndices = np.squeeze(blx*nrows*np.ones((1,nrows)) + range(nrows))
        colIndices = np.squeeze(bly*nrows*np.ones((1,nrows)) + range(nrows))
    block = arr[rowIndices.astype(int),:][:,colIndices.astype(int)]
    return block


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
[nx, nu] = B.shape

#Discretize
Ad = sparse.csc_matrix(sp.linalg.expm(A*T))
x = sy.symbols('x')
Ax = sy.Matrix(A)*x
eAx = Ax.exp()
#eAxF = sy.lambdify((x),eAx)
eAxInt = np.empty([A.shape[0],A.shape[0]])
for (i,j), func_ij in np.ndenumerate(eAx):
    func = sy.lambdify((x),func_ij)
    eAxInt[i,j] = sp.integrate.quad(func,0.,T)[0]
Bd = sparse.csc_matrix(eAxInt@B)
# Ad1 = Ad.toarray()
# Ad2 = eAxF(T)
# AdD = Ad2-Ad1

# #Testing discretized system
# x0 = np.zeros(nx)
# tFin = 10
# dt = 0.01
# tCont = np.arange(0,tFin+dt,dt)
# tDisc = np.arange(0,tFin+T,T)
# xCont = np.empty([nx,tCont.shape[0]])
# xCont[:,0] = x0
# xDisc= np.empty([nx,tDisc.shape[0]])
# xDisc[:,0] = x0
# u = np.array([1,1])

# for i in range(1,tCont.shape[0]):
#     xCont[:,i] = xCont[:,i-1] + (A@xCont[:,i-1] + B@u)*dt

# for i in range(1,tDisc.shape[0]):
#     xDisc[:,i] = Ad@xDisc[:,i-1] + Bd@u

# plt.figure(1)
# plt.plot(tCont,xCont[0,:])
# plt.plot(tDisc,xDisc[0,:])
# plt.show()

# Constraints
umin = np.array([-0.2, -0.2])
umax = np.array([0.2, 0.2])

# Objective function
Q = 2e+03*sparse.diags([2**2., 11.3**2., 0.01, 10])
R = sparse.diags([2.14, 1])
Q = 5e+03*sparse.diags([2.15**2., 11.3**2., 0.01, 200])
R = sparse.diags([2.14, 0.0001])
Q = 5e+03*sparse.diags([2.15**2., 11.3**2., 0.01, 200])
R = sparse.diags([4.5, 0.0001])
Q = 5e+03*sparse.diags([2.15**2., 11.3**2., 0.0001, 200])
R = sparse.diags([4.5, 0.0001])
Q = 5e+03*sparse.diags([2.15**2., 11.3**2., 0.0001, 150])
R = sparse.diags([4.5, 0.0001])
Q = 1.5e+03*sparse.diags([2**2., 11**2., 0.0001, 200])
R = 0.7**2*sparse.eye(2)
Q = 1.5e+03*sparse.diags([2**2., 11**2., 0.0001, 200])
R = sparse.diags([0.1, 0.001])

Q = 8e+02*sparse.diags([2**2., 11**2., 0.0001, 900])
R = 1000**2*sparse.diags([1, 1])
Q = 8e+02*sparse.diags([2**2., 10**2., 0.0001, 900])
R = 1000**2*sparse.diags([1, 1])
Q = 8e+02*sparse.diags([0.085**2., 10**2., 0.0001, 900])
R = 1000**2*sparse.diags([1, 1])
Q = 8e+02*sparse.diags([0.2**2., 10**2., 3.8**2, 900])
R = 1000**2*sparse.diags([1, 1])


#LQR Stuff
S = sp.linalg.solve_discrete_are(Ad.toarray(), Bd.toarray(), Q.toarray(), R.toarray())
K = np.asarray(np.linalg.inv(R + np.transpose(Bd)@S@Bd)@(np.transpose(Bd)@S@Ad))
QN = S

if ~(np.all(np.linalg.eigvals(S) > 0)):
    raise Exception("Riccati solution not positive definite")

# Initial and reference states
rx = rp
ry = 0
x0 = np.array([100.,10,0.,0.])
xr = np.array([rx,ry,0.,0.])

# Horizons
Nx = 40
Nc = 5
Nb = 5

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(Nx), Q), QN,
                       sparse.kron(sparse.eye(Nc), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(Nx), -Q@xr), -QN@xr, np.zeros(Nc*nu)])
# - linear dynamics
Ax1 = sparse.kron(sparse.eye(Nc+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(Nc+1, k=-1), Ad)
Ax2 = sparse.kron(sparse.eye(Nx-Nc),-sparse.eye(nx)) + sparse.kron(sparse.eye(Nx-Nc, k=-1), (Ad-Bd@K))
Ax3 = sparse.block_diag([Ax1, Ax2], format='csr')
Ax4 = sparse.csr_matrix((Nx+1,Nx+1))
Ax4[Nc+1,Nc] = 1
Ax4 = sparse.kron(Ax4, (Ad-Bd@K))
Ax = Ax3 + Ax4
BuI = sparse.vstack([sparse.csc_matrix((1, Nc)), sparse.eye(Nc), sparse.csc_matrix((Nx-Nc, Nc))])
Bu = sparse.kron(BuI, Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(Nx*nx)])
ueq = leq
# - input and state constraints
C_11 = math.sin(phi+gam)/((rp-rtot)*math.sin(gam))
C_12 = -math.cos(phi+gam)/((rp-rtot)*math.sin(gam))
C_21 = -math.sin(phi-gam)/((rp-rtot)*math.sin(gam))
C_22 = math.cos(phi-gam)/((rp-rtot)*math.sin(gam))
C = np.array([
            [C_11, C_12, 0., 0.],
            [C_21, C_22, 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 1.,],
             ])
ny = C.shape[0]
Aineq1 = sparse.kron(sparse.eye(Nx+1), C)
Aineq2 = sparse.kron(sparse.eye(Nc), sparse.eye(nu))
Aineq = sparse.block_diag([Aineq1, Aineq2], format='dia')
xmin = np.array([1., 1., rp, 0.])
xmax = np.array([np.inf, np.inf, np.inf, np.absolute(x0[0]-rx) + np.absolute(x0[1]-ry)])
lineq = np.hstack([np.kron(np.ones(Nb+1), xmin), np.kron(np.ones(Nx-Nb),-np.inf*np.ones(ny)), np.kron(np.ones(Nc), umin)])
uineq = np.hstack([np.kron(np.ones(Nb+1), xmax), np.kron(np.ones(Nx-Nb), np.inf*np.ones(ny)), np.kron(np.ones(Nc), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

# Simulate in closed loop
nsim = 1000
ifail = nsim
xk = np.empty([nx,nsim+1])
xv1n = np.empty([1,nsim+1])
noiseStored = np.empty([nx,nsim+1])
ctrls = np.empty([nu,nsim])
xk[:,0] = x0
xv1n[0,0] = xk[2,0] + xk[3,0]
noiseVec = sigMat@random.normal(0, 1, 4)
noiseStored[:,0] = noiseVec
for i in range(nsim):

    # Solve
    res = prob.solve()

    #Check solver status
    if res.info.status != 'solved':
       ifail = i
       break
       #raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[(Nx+1)*nx:(Nx+1)*nx+nu]
    ctrls[:,i] = ctrl
    x0 = Ad@x0 + Bd@ctrl + noiseVec
    xk[:,i+1] = x0
    xv1n[0,i+1] = np.absolute(x0[2]) + np.absolute(x0[3])

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

    #Reconfigure velocity constraint
    C1 = (-1, 1)[x0[2] >= 0]
    C2 = (-1, 1)[x0[3] >= 0]
    # gam1 = 45*(np.pi/180)
    # C_11 = math.sin(phi+gam1)/((rp-rtot)*math.sin(gam1))
    # C_12 = -math.cos(phi+gam1)/((rp-rtot)*math.sin(gam1))
    # C_21 = -math.sin(phi-gam1)/((rp-rtot)*math.sin(gam1))
    # C_22 = math.cos(phi-gam1)/((rp-rtot)*math.sin(gam1))
    C = np.array([
            [C_11, C_12, 0., 0.],
            [C_21, C_22, 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., C1, C2],
             ])
    Aineq1 = sparse.kron(sparse.eye(Nx+1), C)
    Aineq2 = sparse.kron(sparse.eye(Nc), sparse.eye(nu))
    Aineq = sparse.block_diag([Aineq1, Aineq2], format='dia')
    A = sparse.vstack([Aeq, Aineq], format='csc')
    xmin = np.array([1., 1., rp, 0.])
    xmax = np.array([np.inf, np.inf, np.inf, np.absolute(x0[0]-rx) + np.absolute(x0[1]-ry)])
    lineq = np.hstack([np.kron(np.ones(Nb+1), xmin), np.kron(np.ones(Nx-Nb),-np.inf*np.ones(ny)), np.kron(np.ones(Nc), umin)])
    uineq = np.hstack([np.kron(np.ones(Nb+1), xmax), np.kron(np.ones(Nx-Nb), np.inf*np.ones(ny)), np.kron(np.ones(Nc), umax)])
    l[(Nx+1)*nx:] = lineq
    u[(Nx+1)*nx:] = uineq
    prob.update(Ax = A.data, l=l, u=u)

    #Inject noise into plant
    noiseVec = sigMat@random.normal(0, 1, 4)
    noiseStored[:,i+1] = noiseVec

print(ifail)

#Plotting Constraints and Obstacles
xInt = 0.1
xSamps = np.arange(0,110,xInt)
yVertSamps = np.arange(-10,10+xInt,xInt)
xVertSamps = np.ones(yVertSamps.shape)
yConeL = ((rp-rtot)*math.sin(gam)/(math.cos(phi-gam))) + math.tan(phi-gam)*xSamps
yConeU = -((rp-rtot)*math.sin(gam)/(math.cos(phi+gam))) + math.tan(phi+gam)*xSamps
#yRotTan = rp*math.sin(gam)/math.sin(phi) - math.cos(phi)/math.sin(phi)*xSamps
vertCons = rp*math.sin(gam)
vertCons = rp
xVertSamps = xVertSamps*vertCons
xCirc = np.arange(-rp,rp+xInt,xInt)
xCircSq = np.square(xCirc)
topCircle = np.sqrt(rp**2-np.round(xCircSq,2))
botCircle = -np.sqrt(rp**2-np.round(xCircSq,2))


plt.figure(1)
plt.plot(xCirc,topCircle)
plt.plot(xCirc,botCircle)
plt.plot(xSamps,yConeL)
plt.plot(xSamps,yConeU)
plt.plot(xVertSamps,yVertSamps)
plt.plot(xk[0,:ifail+1],xk[1,:ifail+1])
#plt.plot(xSamps,yRotTan)
ax = plt.gca()
ax.set_aspect('equal')


plt.figure(2)
x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3, sharey = x3p)

xTime = [T*x for x in range(ifail+1)]
x1p.plot(xTime,xk[0,:ifail+1])
x2p.plot(xTime,xk[1,:ifail+1])
x3p.plot(xTime,xk[2,:ifail+1])
x4p.plot(xTime,xk[3,:ifail+1])


plt.figure(3)
u1p = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=3)
u2p = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=3)


uTime = [T*x for x in range(1,ifail+1)]
u1p.plot(uTime,ctrls[0,:ifail])
u2p.plot(uTime,ctrls[1,:ifail])


plt.figure(4)
plt.plot(np.abs(xk[0,:ifail+1]-rx)+np.abs(xk[1,:ifail+1]-ry),np.abs(xk[0,:ifail+1]-rx)+np.abs(xk[1,:ifail+1]-ry))
plt.plot(np.abs(xk[0,:ifail+1]-rx)+np.abs(xk[1,:ifail+1]-ry),np.reshape(xv1n[0,:ifail+1], ifail+1))
plt.show()
