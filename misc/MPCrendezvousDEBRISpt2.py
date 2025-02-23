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

#Debris bounding box
center = np.array([40,0])
sideLength = 5
sqVerts = np.array([[center[0]+sideLength/2, center[1]+sideLength/2],
                    [center[0]-sideLength/2, center[1]+sideLength/2],
                    [center[0]-sideLength/2, center[1]-sideLength/2],
                    [center[0]+sideLength/2, center[1]-sideLength/2]])

# Constraints
V1min = 1
umin = np.array([-0.2, -0.2])
umax = np.array([0.2, 0.2])
xmin = np.array([1., 1., rp, 0.,0])
xmax = np.array([np.inf, np.inf, np.inf, V1min, np.inf])

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

#new, untested
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
x0 = np.array([100.,16.5,0.,0.])
xr = np.array([rx,ry,0.,0.])

# Horizons
Nx = 20
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
slope = (x0[1]-sqVerts[0,1])/(x0[0]-sqVerts[0,0])
inter = -slope*x0[0] + x0[1]
C = np.array([
        [C_11, C_12, 0., 0.],
        [C_21, C_22, 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 1., 1.],
        [-slope, 1., 0., 0.],
             ])
ny = C.shape[0]
Aineq1 = sparse.kron(sparse.eye(Nx+1), C)
Aineq2 = sparse.kron(sparse.eye(Nc), sparse.eye(nu))
Aineq = sparse.block_diag([Aineq1, Aineq2], format='dia')
xmin = np.array([1., 1., rp, 0.,-np.inf])
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
nsim = 1200
xk = np.empty([nx,nsim+1])
xv1n = np.empty([1,nsim+1])
xk[:,0] = x0
xv1n[0,0] = xk[2,0] + xk[3,0]
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
    xv1n[0,i+1] = np.absolute(x0[2]) + np.absolute(x0[3])

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

    #Reconfigure velocity constraint
    C1 = (-1, 1)[x0[2] >= 0]
    C2 = (-1, 1)[x0[3] >= 0]
    if (x0[0] - (center[0] + sideLength/2) < 0 and x0[0] - (center[0] - sideLength/2) > 0):
        slope = (x0[1]-sqVerts[1,1])/(x0[0]-sqVerts[1,0])
        inter = -slope*x0[0] + x0[1]
    else:
        slope = (x0[1]-sqVerts[0,1])/(x0[0]-sqVerts[0,0])
        inter = -slope*x0[0] + x0[1]
    C = np.array([
            [C_11, C_12, 0., 0.],
            [C_21, C_22, 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., C1, C2],
            [-slope, 1., 0., 0.],
             ])
    Aineq1 = sparse.kron(sparse.eye(Nx+1), C)
    Aineq2 = sparse.kron(sparse.eye(Nc), sparse.eye(nu))
    Aineq = sparse.block_diag([Aineq1, Aineq2], format='dia')
    A = sparse.vstack([Aeq, Aineq], format='csc')
    if (x0[0] - (center[0] + sideLength/2) < 0 and x0[0] - (center[0] - sideLength/2) > 0):
        xmin = np.array([1., 1., rp, 0., inter])
    elif (x0[0] - (center[0] + sideLength/2) < 20 and x0[0] - (center[0] + sideLength/2) > 0):
        xmin = np.array([1., 1., rp, 0., inter])
    else:
        xmin = np.array([1., 1., rp, 0., -np.inf])
    xmax = np.array([np.inf, np.inf, np.inf, np.absolute(x0[0]) + np.absolute(x0[1]), np.inf])
    lineq = np.hstack([np.kron(np.ones(Nb+1), xmin), np.kron(np.ones(Nx-Nb),-np.inf*np.ones(ny)), np.kron(np.ones(Nc), umin)])
    uineq = np.hstack([np.kron(np.ones(Nb+1), xmax), np.kron(np.ones(Nx-Nb), np.inf*np.ones(ny)), np.kron(np.ones(Nc), umax)])
    l[(Nx+1)*nx:] = lineq
    u[(Nx+1)*nx:] = uineq
    prob.update(Ax = A.data, l=l, u=u)

#Plotting Constraints
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


# #Plot obstacle constraint line
# testPoint = np.array([100,-20])
# slope = (testPoint[1]-sqVerts[0,1])/(testPoint[0]-sqVerts[0,0])
# inter = -slope*testPoint[0] + testPoint[1]
# xSampCon = np.arange(20,100,xInt)
# yDebCon = slope*xSampCon + inter

plt.figure(1)
plt.plot(np.array([sqVerts[1,0],sqVerts[0,0]]),np.array([sqVerts[1,1],sqVerts[0,1]]))
plt.plot(np.array([sqVerts[2,0],sqVerts[3,0]]),np.array([sqVerts[2,1],sqVerts[3,1]]))
plt.plot(np.array([sqVerts[2,0],sqVerts[2,0]]),np.array([sqVerts[2,1],sqVerts[1,1]]))
plt.plot(np.array([sqVerts[3,0],sqVerts[3,0]]),np.array([sqVerts[3,1],sqVerts[0,1]]))
# plt.plot(testPoint[0],testPoint[1],"xr")
# plt.plot(xSampCon,yDebCon)
plt.plot(xCirc,topCircle)
plt.plot(xCirc,botCircle)
plt.plot(xSamps,yConeL)
plt.plot(xSamps,yConeU)
plt.plot(xVertSamps,yVertSamps)
plt.plot(xk[0,:],xk[1,:])
#plt.plot(xSamps,yRotTan)
ax = plt.gca()
ax.set_aspect('equal')
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