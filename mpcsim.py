from typing import (Tuple, Any)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

class Noise:
    def __init__(self, noise_std:Tuple[float,float], noise_length:float):
        self.noise_std = noise_std
        self.noise_length = noise_length
    def constructSigMat(self):
        sigMat = np.diag([self.noise_std[0], self.noise_std[1], 0, 0])
        return sigMat


class SimConditions:
    def __init__(self, x0:np.ndarray[tuple[int, ...], 'float64'], xr:np.ndarray[tuple[int, ...], 'float64'], r_p:float, los_ang:float, r_tol:float, hatch_ofst:float, mean_mtn:float, time_stp:float, isReject:bool, suc_cond:Tuple[float,float], noise:Noise):
        self.x0 = x0
        self.xr = xr
        self.r_p = r_p
        self.los_ang = los_ang
        self.r_tol = r_tol
        self.hatch_ofst = hatch_ofst
        self.mean_mtn = mean_mtn
        self.time_stp = time_stp
        self.isReject = isReject
        self.suc_cond = suc_cond
        self.noise = noise

class SimRun:
    def __init__(self, i_term:int, isSuccess:bool, x_true_pcw, x_est, ctrl_hist, ctrlr_seq, noise_hist):
        self.i_term = i_term
        self.isSuccess = isSuccess
        self.x_true_pcw = x_true_pcw
        self.x_est = x_est
        self.ctrl_hist = ctrl_hist
        self.ctrlr_seq = ctrlr_seq
        self.noiseHist = noise_hist

class Debris:
    def __init__(self, center:Tuple[float,float], side_length:float):
        self.center = center
        self.side_length = side_length
    def constructVertArr(self):
        sqVerts = np.array([[self.center[0] + self.side_length / 2, self.center[1] + self.side_length / 2],
                            [self.center[0] - self.side_length / 2, self.center[1] + self.side_length / 2],
                            [self.center[0] - self.side_length / 2, self.center[1] - self.side_length / 2],
                            [self.center[0] + self.side_length / 2, self.center[1] - self.side_length / 2]])
        return sqVerts



class MPCParams:
    def __init__(self, Q_state, R_input, R_slack, V_ecr, horizons):
        self.Q_state = Q_state
        self.R_input = R_input
        self.R_slack = R_slack
        self.V_ecr = V_ecr
        self.Nx = horizons["Nx"]
        self.Nc = horizons["Nc"]
        self.Nb = horizons["Nb"]


class FailsafeParams:
    def __init__(self, Q_fail, R_fail, C_int, K_dead):
        self.Q_fail = Q_fail
        self.R_fail = R_fail
        self.C_int = C_int
        self.K_dead = K_dead


def figurePlotSave(sim_conditions:SimConditions, debris:Debris, sim_run:SimRun, saveCounter=None):

    xtruePiece = sim_run.x_true_pcw
    xestO = sim_run.x_est
    noiseStored = sim_run.noiseHist
    ctrls = sim_run.ctrl_hist
    iterm = sim_run.i_term
    controllerSeq = sim_run.ctrlr_seq


    def numberToColor(num):
        if (num == 1):
            col = 'b'
        elif (num == 2):
            col = 'r'
        elif (num == 3):
            col = 'y'
        return col

    # Simulation Constants
    gam = sim_conditions.los_ang
    rp = sim_conditions.r_p
    rtot = sim_conditions.r_tol
    phi = sim_conditions.hatch_ofst
    n = sim_conditions.mean_mtn
    T = sim_conditions.time_stp

    rx = sim_conditions.xr[0]
    ry = sim_conditions.xr[1]

    # Debris bounding box
    if debris is not None:
        sqVerts = debris.constructVertArr()

    xInt = 0.1
    xSamps = np.arange(0, 110, xInt)
    xTime = [T * x for x in range(iterm)]

    # contruct velocity one norms
    xv1n = np.empty(xtruePiece.shape[1])
    for i in range(xtruePiece.shape[1]):
        xv1n[i] = np.absolute(xtruePiece[2, i]) + np.absolute(xtruePiece[3, i])

    # Plotting Constraints and Obstacles
    yVertSamps = np.arange(-10, 10 + xInt, xInt)
    xVertSamps = np.ones(yVertSamps.shape)
    yConeL = ((rp - rtot) * math.sin(gam) / (math.cos(phi - gam))) + math.tan(phi - gam) * xSamps
    yConeU = -((rp - rtot) * math.sin(gam) / (math.cos(phi + gam))) + math.tan(phi + gam) * xSamps
    vertCons = rp * math.sin(gam)
    vertCons = rp
    xVertSamps = xVertSamps * vertCons
    xCirc = np.arange(-rp, rp + xInt, xInt)
    xCircSq = np.square(xCirc)
    topCircle = np.sqrt(rp ** 2 - np.round(xCircSq, 2))
    botCircle = -np.sqrt(rp ** 2 - np.round(xCircSq, 2))



    ConsComb, (geoConp, velConp) = plt.subplots(nrows=2, ncols=1)
    ConsComb.set_size_inches((5,5.5))
    ConsComb.set_dpi(300)

    if debris is not None:
        geoConp.plot(np.array([sqVerts[1, 0], sqVerts[0, 0]]), np.array([sqVerts[1, 1], sqVerts[0, 1]]), color='#994F00')
        geoConp.plot(np.array([sqVerts[1, 0], sqVerts[0, 0]]), np.array([sqVerts[1, 1], sqVerts[0, 1]]), color='#994F00')
        geoConp.plot(np.array([sqVerts[2, 0], sqVerts[3, 0]]), np.array([sqVerts[2, 1], sqVerts[3, 1]]), color='#994F00')
        geoConp.plot(np.array([sqVerts[2, 0], sqVerts[2, 0]]), np.array([sqVerts[2, 1], sqVerts[1, 1]]), color='#994F00')
        geoConp.plot(np.array([sqVerts[3, 0], sqVerts[3, 0]]), np.array([sqVerts[3, 1], sqVerts[0, 1]]), color='#994F00')

    geoConp.plot(xCirc, topCircle, color='0.5')
    geoConp.plot(xCirc, botCircle, color='0.5')
    geoConp.plot(xSamps, yConeL, color='#994F00')
    geoConp.plot(xSamps, yConeU, color='#994F00')
    geoConp.plot(xVertSamps, yVertSamps, color='#994F00')
    for i in range(iterm - 1):
        geoConp.plot(xtruePiece[0, i:i + 2], xtruePiece[1, i:i + 2], color=numberToColor(controllerSeq[i + 1]))
    customLines = [Line2D([0], [0], color='b'),
                   Line2D([0], [0], color='r'),
                   Line2D([0], [0], color='y')]
    geoConp.set_aspect('equal')
    geoConp.title.set_text('Trajectory and Contraints (LVLH)')
    geoConp.set_ylabel('$\mathregular{\delta}$y (m)')
    geoConp.set_xlabel('$\mathregular{\delta}$x (m)')
    geoConp.legend(customLines, ['MPC Controller', 'LQR Failsafe', 'LQR Debris Avoidance'], loc='lower right',
                   prop={'size': 5})
    velConp.set_xlabel('Relative Position L1 Norm (m)')
    velConp.set_ylabel('Relative Position L1 Norm (m)')
    velConp.plot(np.abs(xtruePiece[0, :iterm + 1] - rx) + np.abs(xtruePiece[1, :iterm + 1] - ry),
                 np.abs(xtruePiece[0, :iterm + 1] - rx) + np.abs(xtruePiece[1, :iterm + 1] - ry), color='#994F00',
                 label='_nolegend_')
    velConp.plot(np.abs(xtruePiece[0, :iterm + 1] - rx) + np.abs(xtruePiece[1, :iterm + 1] - ry),
                 np.reshape(xv1n[:iterm], iterm), color='b', label='Relative Velocity L1 Norm')
    velConp.legend(['Relative Velocity L1 Norm (m/s)'], loc='upper left', prop={'size': 5})

    estTrueStates = plt.figure(2)
    x1p = plt.subplot2grid((6, 3), (0, 0), rowspan=1, colspan=3)
    x2p = plt.subplot2grid((6, 3), (1, 0), rowspan=1, colspan=3)
    x3p = plt.subplot2grid((6, 3), (2, 0), rowspan=1, colspan=3)
    x4p = plt.subplot2grid((6, 3), (3, 0), rowspan=1, colspan=3)
    d1p = plt.subplot2grid((6, 3), (4, 0), rowspan=1, colspan=3)
    d2p = plt.subplot2grid((6, 3), (5, 0), rowspan=1, colspan=3)

    x1p.plot(xTime, xtruePiece[0, :iterm + 1])
    x1p.plot(xTime, xestO[0, :iterm])
    x2p.plot(xTime, xtruePiece[1, :iterm + 1])
    x2p.plot(xTime, xestO[1, :iterm])
    x3p.plot(xTime, xtruePiece[2, :iterm + 1])
    x3p.plot(xTime, xestO[2, :iterm])
    x4p.plot(xTime, xtruePiece[3, :iterm + 1])
    x4p.plot(xTime, xestO[3, :iterm])
    d1p.plot(xTime, noiseStored[0, :iterm])
    d1p.plot(xTime, xestO[4, :iterm])
    d2p.plot(xTime, noiseStored[1, :iterm])
    d2p.plot(xTime, xestO[5, :iterm])

    estTrueStates.set_size_inches((7, 7.5))
    estTrueStates.set_dpi(300)

    x1p.title.set_text('True and Estimated States (LVLH)')
    x1p.set_ylabel('$\mathregular{\delta}$x (m)')
    x1p.legend(['True','Estimated'], loc='upper right')
    x1p.xaxis.set_visible(False)
    x2p.set_ylabel('$\mathregular{\delta}$y (m)')
    x2p.xaxis.set_visible(False)
    x3p.set_ylabel('$\mathregular{\delta\dot{x}}$ (m/s)')
    x3p.xaxis.set_visible(False)
    x4p.set_ylabel('$\mathregular{\delta\dot{y}}$ (m/s)')
    x4p.xaxis.set_visible(False)
    d1p.set_ylabel('$\mathregular{d_x}$ (m)')
    d1p.xaxis.set_visible(False)
    d2p.set_ylabel('$\mathregular{d_y}$ (m)')
    d2p.set_xlabel('Time (s)')

    estTrueStates.align_labels()

    controlPlot = plt.figure(3)
    u1p = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=3)
    u2p = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=3)

    uTime = [T * x for x in range(1, iterm + 1)]
    u1p.plot(uTime, ctrls[0, :iterm])
    u2p.plot(uTime, ctrls[1, :iterm])

    u1p.title.set_text('Actuator Commands (LVLH)')
    u1p.set_ylabel('$\mathregular{u_x}$ $\mathregular{(m/s^2)}$')
    u2p.set_ylabel('$\mathregular{u_y}$ $\mathregular{(m/s^2)}$')
    u2p.set_xlabel('Time (s)')

    if saveCounter != None:
        iter = str(saveCounter) + '.png'
        direc = 'RunFigs/'
        # TWODtraj.savefig(direc + '2Dtraj' + iter)
        estTrueStates.savefig(direc + 'trueANDest' + iter,dpi=300)
        controlPlot.savefig(direc + 'contrHist' + iter,dpi=300)
        # velCon.savefig(direc + 'velConstraint' + iter)
        ConsComb.savefig(direc + 'combCons' + iter,dpi=300)
        plt.close('all')
    else:
        plt.show()
        plt.close('all')
