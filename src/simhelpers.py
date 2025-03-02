from scipy import sparse
import numpy as np
from numpy import random
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import control as ct

from src.mpcsim import *

def configureDynamicConstraints(sim_conditions:SimConditions, mpc_params:MPCParams, debris:Debris, xest, block_mats, u_lim):

    # xest = np.copy(xest)

    def debugPlotConstraint(sqVerts,xest,inTrack,slope,intercept):

        def pltline(slope,intercept):
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope*x_vals
            plt.plot(x_vals,y_vals,'--')

        plt.plot(0,0,'bx')
        if (inTrack):
            plt.plot(xest[1],xest[0],'ro')
        else:
            plt.plot(xest[0], xest[1], 'ro')
        plt.plot(np.array([sqVerts[1, 0], sqVerts[0, 0]]), np.array([sqVerts[1, 1], sqVerts[0, 1]]),
                     color='#994F00', label='_nolegend_')
        plt.plot(np.array([sqVerts[1, 0], sqVerts[0, 0]]), np.array([sqVerts[1, 1], sqVerts[0, 1]]),
                     color='#994F00', label='_nolegend_')
        plt.plot(np.array([sqVerts[2, 0], sqVerts[3, 0]]), np.array([sqVerts[2, 1], sqVerts[3, 1]]),
                     color='#994F00', label='_nolegend_')
        plt.plot(np.array([sqVerts[2, 0], sqVerts[2, 0]]), np.array([sqVerts[2, 1], sqVerts[1, 1]]),
                     color='#994F00', label='_nolegend_')
        plt.plot(np.array([sqVerts[3, 0], sqVerts[3, 0]]), np.array([sqVerts[3, 1], sqVerts[0, 1]]),
                     color='#994F00', label='_nolegend_')
        pltline(slope,intercept)
        plt.gca().set_aspect('equal')
        plt.show()

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

    if (sim_conditions.inTrack):
        xestCalc = np.copy(xest)
        xest[0], xest[1] = xest[1], xest[0]
        temp = center
        center = list(center)
        center[0], center[1] = temp[1], temp[0]
    else:
        xestCalc = xest
    if (xest[1] >= 0): #Switch less than, equals signs in if statement to see under behavior at centerline

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

    C[3,2] = C1
    C[3,3] = C2
    C[4,0] = -slope
    Aineq1 = sparse.kron(sparse.eye(Nx + 1), C)
    Aineq = sparse.block_array(([Aineq1, Block12], [Block21, Aineq2]), format='dia')
    A = sparse.vstack([Aeq, Aineq], format='csc')
    A = sparse.hstack([A, AextCol])
    A = sparse.vstack([A, AextRow])

    if (xest[1] >= 0):  # Switch less than, equals signs in if statement to see under behavior at centerline

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

    lineq = np.hstack([np.kron(np.ones(Nb + 1), xmin), np.kron(np.ones(Nx - Nb), -np.inf * np.ones(ny)), np.kron(np.ones(Nc), umin), isReject * xest[4:6]])  # assume 0 est disturbance at start
    uineq = np.hstack([np.kron(np.ones(Nb + 1), xmax), np.kron(np.ones(Nx - Nb), np.inf * np.ones(ny)), np.kron(np.ones(Nc), umax), isReject * xest[4:6]])

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

def continuousAppendIndex(impc, ifailsf, ifailsd, i):

    if (bool(impc) and impc[-1] == i - 1):
        impc.append(i)
    elif (bool(ifailsf) and ifailsf[-1] == i - 1):
        ifailsf.append(i)
    elif (bool(ifailsd) and ifailsd[-1] == i - 1):
        ifailsd.append(i)

def integrateNoise(Ap, Bnoise, Qw, T):
    Aop = sp.linalg.block_diag(Ap,np.zeros([2,2]))
    n = Ap.shape[0] + 2
    phi = np.block([[Aop,Bnoise@Qw@np.transpose(Bnoise)],[np.zeros([n,n]), -Aop]])
    AB = sp.linalg.expm(phi*T) @ np.vstack([np.zeros([n,n]), np.eye(n)])
    Qw = AB[:n,:] * np.linalg.inv(AB[n:2*n,:])
    return Qw

def unscentedKF():
