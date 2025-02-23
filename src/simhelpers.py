from scipy import sparse
import numpy as np

from mpcsim import (Debris, SimConditions, MPCParams)

def configureDynamicConstraints(sim_conditions:SimConditions, mpc_params:MPCParams, debris:Debris, xest, block_mats, u_lim):

    rp = sim_conditions.r_p
    x0 = sim_conditions.x0
    xr = sim_conditions.xr
    isReject = sim_conditions.isReject
    rx = xr[0]
    ry = xr[1]
    Nx = mpc_params.Nx
    Nc = mpc_params.Nc
    Nb = mpc_params.Nb

    Aeq = block_mats[0]
    Aineq2 = block_mats[1]
    Block12 = block_mats[2]
    Block21 = block_mats[3]
    AextRow = block_mats[4]
    AextCol = block_mats[5]
    C = block_mats[6]
    ny = C.shape[0]

    umin = u_lim[0]
    umax = u_lim[1]

    if debris is not None:
        # Debris bounding box
        sqVerts = debris.constructVertArr()

        # delete these eventually
        center = debris.center
        sideLength = debris.side_length
        hasDebris = True
    else:
        center = (-np.inf, -np.inf)
        sideLength = 0
        hasDebris = False

    C1 = (-1, 1)[xest[2] >= 0]
    C2 = (-1, 1)[xest[3] >= 0]
    if (xest[0] - (center[0] + sideLength / 2) < 0 and xest[0] - (center[0] - sideLength / 2) > 0):
        slope = (xest[1] - sqVerts[1, 1]) / (xest[0] - sqVerts[1, 0])
        inter = -slope * xest[0] + xest[1]
    elif (hasDebris):
        slope = (xest[1] - sqVerts[0, 1]) / (xest[0] - sqVerts[0, 0])
        inter = -slope * xest[0] + xest[1]
    else:
        slope = 0
    C[3,2] = C1
    C[3,3] = C2
    C[4,0] = -slope
    Aineq1 = sparse.kron(sparse.eye(Nx + 1), C)
    Aineq = sparse.block_array(([Aineq1, Block12], [Block21, Aineq2]), format='dia')
    A = sparse.vstack([Aeq, Aineq], format='csc')
    A = sparse.hstack([A, AextCol])
    A = sparse.vstack([A, AextRow])
    if (xest[0] - (center[0] + sideLength / 2) < 0 and xest[0] - (center[0] - sideLength / 2) > 0):
        xmin = np.array([1., 1., rp, 0., inter])
    elif (xest[0] - (center[0] + sideLength / 2) < 20 and xest[0] - (center[0] + sideLength / 2) > 0):
        xmin = np.array([1., 1., rp, 0., inter])
    else:
        xmin = np.array([1., 1., rp, 0., -np.inf])
    xmax = np.array([np.inf, np.inf, np.inf, np.absolute(xest[0] - rx) + np.absolute(xest[1] - ry), np.inf])
    lineq = np.hstack(
        [np.kron(np.ones(Nb + 1), xmin), np.kron(np.ones(Nx - Nb), -np.inf * np.ones(ny)), np.kron(np.ones(Nc), umin),
         isReject * xest[4:6]])  # assume 0 est disturbance at start
    uineq = np.hstack(
        [np.kron(np.ones(Nb + 1), xmax), np.kron(np.ones(Nx - Nb), np.inf * np.ones(ny)), np.kron(np.ones(Nc), umax),
         isReject * xest[4:6]])

    return A, lineq, uineq

def constructOsqpAeq(mpc_params:MPCParams, Ad, Bd, K, ny):

    nx = Ad.shape[0]

    Nx = mpc_params.Nx
    Nc = mpc_params.Nc
    Nb = mpc_params.Nb

    Ax1 = sparse.kron(sparse.eye(Nc + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(Nc + 1, k=-1), Ad)
    Ax2 = sparse.kron(sparse.eye(Nx - Nc), -sparse.eye(nx)) + sparse.kron(sparse.eye(Nx - Nc, k=-1), (Ad - Bd @ K))
    Ax3 = sparse.block_diag([Ax1, Ax2], format='csr')
    Ax4 = sparse.csr_matrix((Nx + 1, Nx + 1))
    Ax4[Nc + 1, Nc] = 1
    Ax4 = sparse.kron(Ax4, (Ad - Bd @ K))
    Ax = Ax3 + Ax4
    BuI = sparse.vstack([sparse.csc_matrix((1, Nc)), sparse.eye(Nc), sparse.csc_matrix((Nx - Nc, Nc))])
    Bdaug = sparse.hstack([Bd, np.zeros([nx, ny])])
    Bu = sparse.kron(BuI, Bdaug)
    Aeq = sparse.hstack([Ax, Bu])

    return Aeq