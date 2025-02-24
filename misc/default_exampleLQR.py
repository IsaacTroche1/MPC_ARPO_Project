import osqp
import numpy as np
import scipy as sp
import math
from scipy import sparse
from numpy import random
import matplotlib.pyplot as plt

# How to index blocks
# Ax[np.squeeze(1*nx*np.ones((1,12)) + range(nx)),:][:,np.squeeze(1*nx*np.ones((1,12)) + range(nx))].toarray()

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
Ad = sparse.csc_matrix([
  [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
  [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
  [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
  [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
  [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
  [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
  [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
  [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
  [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
])
Bd = sparse.csc_matrix([
  [0.,      -0.0726,  0.,     0.0726],
  [-0.0726,  0.,      0.0726, 0.    ],
  [-0.0152,  0.0152, -0.0152, 0.0152],
  [-0.,     -0.0006, -0.,     0.0006],
  [0.0006,   0.,     -0.0006, 0.0000],
  [0.0106,   0.0106,  0.0106, 0.0106],
  [0,       -1.4512,  0.,     1.4512],
  [-1.4512,  0.,      1.4512, 0.    ],
  [-0.3049,  0.3049, -0.3049, 0.3049],
  [-0.,     -0.0236,  0.,     0.0236],
  [0.0236,   0.,     -0.0236, 0.    ],
  [0.2107,   0.2107,  0.2107, 0.2107]])
[nx, nu] = Bd.shape

# Constraints
u0 = 10.5916
umin = np.array([1., 1., 1., 1.]) - u0
umax = np.array([35., 35., 35., 35.]) - u0
xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
                 -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
xmax = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
                  np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

# Objective function
Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
R = 0.1*sparse.eye(4)

#LQR Stuff
S = sp.linalg.solve_discrete_are(Ad.toarray(), Bd.toarray(), Q.toarray(), R.toarray())
K = np.asarray(np.linalg.inv(R + np.transpose(Bd)@S@Bd)@(np.transpose(Bd)@S@Ad))
QN = S

if ~(np.all(np.linalg.eigvals(S) > 0)):
    raise Exception("Riccati solution not positive definite")

# Initial and reference states
x0 = np.zeros(12)
# x0test = 10*np.random.rand(1,12).squeeze()
# x0 = x0test
xr = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

# Prediction horizon
Nx = 40
Nc = 15

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
Aineq = sparse.eye((Nx+1)*nx + Nc*nu)
lineq = np.hstack([np.kron(np.ones(Nx+1), xmin), np.kron(np.ones(Nc), umin)])
uineq = np.hstack([np.kron(np.ones(Nx+1), xmax), np.kron(np.ones(Nc), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

#debugging
# cool = sparseBlockIndex(Bu,[0,0],12,4)
# cool = sparseBlockIndex(Ax,[0,0],12)

# Simulate in closed loop
nsim = 20
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
    # ctrltest = np.transpose((-K@x0))
    # ctrl = ctrltest
    #x0 = np.transpose(Ad.toarray()@x0) + Bd.toarray()@ctrl
    x0 = Ad@x0 + Bd@ctrl
    xk[:,i+1] = x0

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

x3p = plt.subplot2grid((12, 3), (2, 0), rowspan=1, colspan=3, ylim = [0,1.1])
x1p = plt.subplot2grid((12, 3), (0, 0), rowspan=1, colspan=3, sharey = x3p)
x2p = plt.subplot2grid((12, 3), (1, 0), rowspan=1, colspan=3, sharey = x3p)
x4p = plt.subplot2grid((12, 3), (3, 0), rowspan=1, colspan=3, sharey = x3p)
x5p = plt.subplot2grid((12, 3), (4, 0), rowspan=1, colspan=3, sharey = x3p)
x6p = plt.subplot2grid((12, 3), (5, 0), rowspan=1, colspan=3, sharey = x3p)
x7p = plt.subplot2grid((12, 3), (6, 0), rowspan=1, colspan=3, sharey = x3p)
x8p = plt.subplot2grid((12, 3), (7, 0), rowspan=1, colspan=3, sharey = x3p)
x9p = plt.subplot2grid((12, 3), (8, 0), rowspan=1, colspan=3, sharey = x3p)
x10p = plt.subplot2grid((12, 3), (9, 0), rowspan=1, colspan=3, sharey = x3p)
x11p = plt.subplot2grid((12, 3), (10, 0), rowspan=1, colspan=3, sharey = x3p)
x12p = plt.subplot2grid((12, 3), (11, 0), rowspan=1, colspan=3, sharey = x3p)

# x3p = plt.subplot2grid((12, 3), (2, 0), rowspan=1, colspan=3)
# x1p = plt.subplot2grid((12, 3), (0, 0), rowspan=1, colspan=3)
# x2p = plt.subplot2grid((12, 3), (1, 0), rowspan=1, colspan=3)
# x4p = plt.subplot2grid((12, 3), (3, 0), rowspan=1, colspan=3)
# x5p = plt.subplot2grid((12, 3), (4, 0), rowspan=1, colspan=3)
# x6p = plt.subplot2grid((12, 3), (5, 0), rowspan=1, colspan=3)
# x7p = plt.subplot2grid((12, 3), (6, 0), rowspan=1, colspan=3)
# x8p = plt.subplot2grid((12, 3), (7, 0), rowspan=1, colspan=3)
# x9p = plt.subplot2grid((12, 3), (8, 0), rowspan=1, colspan=3)
# x10p = plt.subplot2grid((12, 3), (9, 0), rowspan=1, colspan=3)
# x11p = plt.subplot2grid((12, 3), (10, 0), rowspan=1, colspan=3)
# x12p = plt.subplot2grid((12, 3), (11, 0), rowspan=1, colspan=3)

x1p.plot(range(nsim+1),xk[0,:])
x2p.plot(range(nsim+1),xk[1,:])
x3p.plot(range(nsim+1),xk[2,:])
x4p.plot(range(nsim+1),xk[3,:])
x5p.plot(range(nsim+1),xk[4,:])
x6p.plot(range(nsim+1),xk[5,:])
x7p.plot(range(nsim+1),xk[6,:])
x8p.plot(range(nsim+1),xk[7,:])
x9p.plot(range(nsim+1),xk[8,:])
x10p.plot(range(nsim+1),xk[9,:])
x11p.plot(range(nsim+1),xk[10,:])
x12p.plot(range(nsim+1),xk[11,:])
plt.show()