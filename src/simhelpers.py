"""
This module contains various helper function for use in simulation to aid readability/conciseness.
"""

from numpy import random
import scipy as sp
import control as ct

from src.mpcsim import *

def configureDynamicConstraints(sim_conditions:SimConditions, mpc_params:MPCParams, debris:Debris, xest, block_mats, u_lim):
    """
    Dynamically reconfigure constraints for the MPC algorithm during simulation runtime

    :param sim_conditions: A SimConditions object representing general simulation conditions (initial state, orbital parameters, etc.)
    :param mpc_params: An MPCParams object containing the tunable parameters of the MPC controller to be used during simulation
    :param debris: A Debris object containing information describing the debris to be avoided by the control algorithm during simulation
    :param xest: Current state and disturbance estimate
    :param block_mats: Relevant matrices for construction of the QP constraint matrix A
    :param u_lim: Input limits
    :return: QP problem constraint matrix A, upper and lower limit vectors u and l
    """

    # Unpack relevant parameters
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

    # Handle debris and no debris cases
    if (debris is not None):
        # Debris bounding box
        sqVerts = debris.constructVertArr()
        if (sim_conditions.inTrack):
            #Turn debris bounding box on side
            sqVertsO = np.copy(sqVerts)
            sqVerts[0,:], sqVerts[1,:], sqVerts[2,:], sqVerts[3,:] = np.copy(sqVerts[1,:]), np.copy(sqVerts[2,:]), np.copy(sqVerts[3,:]), np.copy(sqVerts[0,:])
        # delete these eventually
        center = debris.center
        sideLength = debris.side_length
        hasDebris = True
        detect_dist = debris.detect_distance
    else:
        center = (-np.inf, -np.inf)
        sideLength = 0
        hasDebris = False
        detect_dist = np.inf

    C1 = (-1, 1)[xest[2] >= 0]
    C2 = (-1, 1)[xest[3] >= 0]

    # Handle in-track and radial IC cases
    if (sim_conditions.inTrack):
        xestCalc = np.copy(xest)
        xest[0], xest[1] = xest[1], xest[0]
        temp = center
        center = list(center)
        center[0], center[1] = temp[1], temp[0]
    else:
        xestCalc = xest

    # Determine if below or above debris to choose trajectory around it
    if (xest[1] >= 0):

        if (xest[0] - (center[0] + sideLength / 2) < 0 and xest[0] - (center[0] - sideLength / 2) > 0):
            slope = (xestCalc[1] - sqVerts[1, 1]) / (xestCalc[0] - sqVerts[1, 0])
            inter = -slope * xestCalc[0] + xestCalc[1]
        elif (hasDebris):
            slope = (xestCalc[1] - sqVerts[0, 1]) / (xestCalc[0] - sqVerts[0, 0])
            inter = -slope * xestCalc[0] + xestCalc[1]
        else:

            slope = 0

    elif (xest[1] < 0):

        if (xest[0] - (center[0] + sideLength / 2) < 0 and xest[0] - (center[0] - sideLength / 2) > 0):
            slope = (xestCalc[1] - sqVerts[2, 1]) / (xestCalc[0] - sqVerts[2, 0])
            inter = -slope * xestCalc[0] + xestCalc[1]

        elif (hasDebris):
            slope = (xestCalc[1] - sqVerts[3, 1]) / (xestCalc[0] - sqVerts[3, 0])
            inter = -slope * xestCalc[0] + xestCalc[1]

        else:
            slope = 0

    # Reconfigure constraint matrix C amd insert into relevant place in QP problem A matrix
    C[3,2] = C1
    C[3,3] = C2
    C[4,0] = -slope
    Aineq1 = sparse.kron(sparse.eye(Nx + 1), C)
    Aineq = sparse.block_array(([Aineq1, Block12], [Block21, Aineq2]), format='dia')
    A = sparse.vstack([Aeq, Aineq], format='csc')
    A = sparse.hstack([A, AextCol])
    A = sparse.vstack([A, AextRow])

    # Determine state constraint limit vectors based on location
    if (xest[1] >= 0):

        if (xest[0] - (center[0] + sideLength / 2) < 0 and xest[0] - (center[0] - sideLength / 2) > 0):
            xmin = np.array([1., 1., rp, 0., inter])
        elif (xest[0] - (center[0] + sideLength / 2) < detect_dist and xest[0] - (center[0] + sideLength / 2) > 0):
            xmin = np.array([1., 1., rp, 0., inter])
        else:
            xmin = np.array([1., 1., rp, 0., -np.inf])
        xmax = np.array([np.inf, np.inf, np.inf, np.absolute(xestCalc[0] - rx) + np.absolute(xestCalc[1] - ry), np.inf])

    elif (xest[1] < 0):

        if (xest[0] - (center[0] + sideLength / 2) < 0 and xest[0] - (center[0] - sideLength / 2) > 0):
            xmax = np.array([np.inf, np.inf, np.inf, np.absolute(xestCalc[0] - rx) + np.absolute(xestCalc[1] - ry), inter])
        elif (xest[0] - (center[0] + sideLength / 2) < detect_dist and xest[0] - (center[0] + sideLength / 2) > 0):
            xmax = np.array([np.inf, np.inf, np.inf, np.absolute(xestCalc[0] - rx) + np.absolute(xestCalc[1] - ry), inter])
        else:
            xmax = np.array([np.inf, np.inf, np.inf, np.absolute(xestCalc[0] - rx) + np.absolute(xestCalc[1] - ry), np.inf])
        xmin = np.array([1., 1., rp, 0., -np.inf])

    # Construct QP problem upper and lower limit vectors u and l
    lineq = np.hstack([np.kron(np.ones(Nb + 1), xmin), np.kron(np.ones(Nx - Nb), -np.inf * np.ones(ny)), np.kron(np.ones(Nc), umin), isReject * xest[4:6]])  # assume 0 est disturbance at start
    uineq = np.hstack([np.kron(np.ones(Nb + 1), xmax), np.kron(np.ones(Nx - Nb), np.inf * np.ones(ny)), np.kron(np.ones(Nc), umax), isReject * xest[4:6]])

    return A, lineq, uineq

def constructOsqpAeq(mpc_params:MPCParams, Ad, Bd, K, ny):
    """
    Construct equality constraint portion of QP problem A matrix

    :param mpc_params:
    :param Ad: Discrete time state-space A matrix
    :param Bd: Discrete time state-space B matrix
    :param K: Virtual LQR controller K matrix
    :param ny: State constraint vector length
    :return: Equality constraint portion of QP problem A matrix
    """

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

def continuousAppendIndex(impc, ifailsf, ifailsd, i):
    """
    Extends controller type categorization to continuous time case

    :param impc: List of previous time steps using the MPC controller
    :param ifailsf: List of previous time steps using the LQR failsafe controller
    :param ifailsd: List of previous time steps using the deadbeat debris avoidance controller
    :param i: Current time step
    """

    if (bool(impc) and impc[-1] == i - 1):
        impc.append(i)
    elif (bool(ifailsf) and ifailsf[-1] == i - 1):
        ifailsf.append(i)
    elif (bool(ifailsd) and ifailsd[-1] == i - 1):
        ifailsd.append(i)


