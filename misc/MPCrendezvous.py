import osqp
import numpy as np
import scipy as sp
import sympy as sy
import math
from scipy import sparse
from numpy import random
import matplotlib.pyplot as plt

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
xmin = np.array([1,1,1])
xmin = np.array([1, 1, rp])
xmax = np.array([np.inf, np.inf, np.inf])

# Objective function
Q = 3e+03*sparse.diags([10**2., 10**2., 1., 1.])
R = 10**2*sparse.eye(2)
Q = 0.015e+03*sparse.diags([9.90**2., 50**2., 0.001, 0.01])
R = 0.1**2*sparse.eye(2)
Q = 1.7e+03*sparse.diags([2**2., 9**2., 0.001, 10])
R = 1**2*sparse.eye(2)

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
x0 = np.array([100.,-16.,0.,0.])
xr = np.array([rx,ry,0.,0.])

# Horizons
Nx = 50
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
C_31 = math.cos(phi)/(rp*math.sin(gam))
C_32 = math.sin(phi)/(rp*math.sin(gam))
C = np.array([
            [C_11, C_12, 0., 0.],
            [C_21, C_22, 0., 0.],
            [C_31, C_32, 0., 0.],
             ])
C = np.array([
            [C_11, C_12, 0., 0.],
            [C_21, C_22, 0., 0.],
            [1., 0., 0., 0.],
             ])
ny = C.shape[0]
Aineq1 = sparse.kron(sparse.eye(Nx+1), C)
Aineq2 = sparse.kron(sparse.eye(Nc), sparse.eye(nu))
Aineq = sparse.block_diag([Aineq1, Aineq2], format='dia')
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
nsim = 2000
xk = np.empty([nx,nsim+1])
xk[:,0] = x0
for i in range(nsim):
    # Solve
    res = prob.solve()

    #Check solver status
    if res.info.status != 'solved':
       raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[(Nx+1)*nx:(Nx+1)*nx+nu]
    x0 = Ad@x0 + Bd@ctrl
    xk[:,i+1] = x0

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

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
plt.plot(xk[0,:],xk[1,:])
#plt.plot(xSamps,yRotTan)
plt.show()

plt.figure(2)
x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3, sharey = x3p)

xTime = [T*x for x in range(nsim+1)]
x1p.plot(xTime,xk[0,:])
x2p.plot(xTime,xk[1,:])
x3p.plot(xTime,xk[2,:])
x4p.plot(xTime,xk[3,:])
plt.show()