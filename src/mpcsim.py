"""
This module contains class definitions for the various objects utilized throughout the simulation for parameter passing
and data organization, as well as a function for the plotting and saving of general results.
"""

import math
from typing import (Tuple, Any)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import sparse

class Noise:
    """
    This class defines the statistical characteristics of the additive output noise applied to the plant
    """
    def __init__(self, noise_std:Tuple[float,float], noise_length:float):
        """
        Constructor for the Noise class
        Args:
            noise_std: Standard deviation of the random x and y disturbances
            noise_length: Length, in control intervals, of randomly generated noise
        """
        self.noise_std = noise_std
        self.noise_length = noise_length
    def constructSigMat(self):
        """
        This method generates matrix of standard deviations for convenience
        :return: Matrix of x and y disturbance standard deviations
        """
        sigMat = np.diag([self.noise_std[0], self.noise_std[1], 0, 0])
        return sigMat


class SimConditions:
    """
    This class defines the general simulation conditions that are independent of the control algorithm
    """
    def __init__(self, x0:np.ndarray[tuple[int, ...], 'float64'], xr:np.ndarray[tuple[int, ...], 'float64'], r_p:float, los_ang:float, r_tol:float, mean_mtn:float, time_stp:float, isReject:bool, suc_cond:Tuple[float,float], noise:Noise=None, inTrack:bool=False, T_cont:float=float('nan'), T_final:int=100, isDeltaV:bool=False):
        """
        Constructor for the SimConditions class
        Args:
            x0: Initial conditions for simulation, in SI units
            xr: Target position for simulation, in SI units
            r_p: Radius of target platform in meters
            los_ang: LOS cone half-angle in radians
            r_tol: LOS cone inlay distance in meters
            hatch_ofst: Angle between LVLH x-axis and docking hatch, in radians
            mean_mtn: Mean motion of target orbit in rad/s
            time_stp: MPC algortihm control interval in seconds
            isReject: Boolean that determines if offset-free method is used to reject disturbances
            suc_cond: Tolerance conditions that determine if simulation was successful, in SI units
            noise: Noise object describing additive output noise to plant
            inTrack: Boolean that signals if the ICs are in the in-track direction
            T_cont: Continuous-time simulation time step in seconds
            T_final: Maximum simulation time in seconds
            isDeltaV: Boolean that determines whether the simulation uses an impulsive delta-v input model
        """
        self.x0 = x0
        self.xr = xr
        self.r_p = r_p
        self.los_ang = los_ang
        self.r_tol = r_tol
        self.hatch_ofst = (inTrack*90)*(np.pi/180)
        self.mean_mtn = mean_mtn
        self.time_stp = time_stp
        self.isReject = isReject
        self.suc_cond = suc_cond
        self.noise = noise
        self.inTrack = inTrack
        self.T_cont = T_cont
        self.T_final = T_final
        self.isDeltaV = isDeltaV

class SimRun:
    """
    This class encapsulates the necessary simulation telemetry for plotting and reduction
    """
    def __init__(self, i_term:int, isSuccess:bool, x_true_pcw, x_est, ctrl_hist, ctrlr_seq, noise_hist):
        """
        Constructor for the SimRun class
        Args:
            i_term: Time step at which the simulation terminated
            isSuccess: Boolean that represents if the simulation was successful based on the provided conditions
            x_true_pcw: Ground truth state trajectories
            x_est: Estimated state trajectories and disturbances
            ctrl_hist: History of control commands
            ctrlr_seq: History of controllers used as an array (see trajectorySimulate.py for definitions)
            noise_hist: History of x and y additive output disturbances
        """
        self.i_term = i_term
        self.isSuccess = isSuccess
        self.x_true_pcw = x_true_pcw
        self.x_est = x_est
        self.ctrl_hist = ctrl_hist
        self.ctrlr_seq = ctrlr_seq
        self.noise_hist = noise_hist

class Debris:
    """
    This class describes the geometry of the debris to be avoided during simulation
    """
    def __init__(self, center:Tuple[float,float], side_length:float, detect_distance:float):
        """
        Constructor for the Debris class
        Args:
            center: LVLH location of the center of the debris bounding box in meters
            side_length: Side length of the debris bounding box in meters
            detect_distance: Distance the algorithm begins to avoid the debris at
        """
        self.center = center
        self.side_length = side_length
        self.detect_distance = detect_distance
    def constructVertArr(self):
        """
        Convenience method that constructs an array representing the vertex locations of the debris bounding box
        :return: Array representing the vertex locations of the debris bounding box
        """
        sqVerts = np.array([[self.center[0] + self.side_length / 2, self.center[1] + self.side_length / 2],
                            [self.center[0] - self.side_length / 2, self.center[1] + self.side_length / 2],
                            [self.center[0] - self.side_length / 2, self.center[1] - self.side_length / 2],
                            [self.center[0] + self.side_length / 2, self.center[1] - self.side_length / 2]])
        return sqVerts



class MPCParams:
    """
    This class encapsulates the tunable parameters of the MPC controller utilized in simulation
    """
    def __init__(self, Q_state, R_input, R_slack, V_ecr, horizons, u_lim:Tuple[float,float], swap_xy:bool=False):
        """
        Constructor for the MPCParams class
        Args:
            Q_state: Penalty matrix for states
            R_input: Penalty matrix for inputs
            R_slack: Penalty matrix for slack variables
            V_ecr: Scaling factors for slack variables
            horizons: Dictionary containing prediction, control, and constrain horizons
            u_lim: Input constraints
            swap_xy: Convenience parameter that swaps Q,R entries to account for runs with in-track initial conditions
        """
        self.Q_state = Q_state
        self.R_input = R_input
        if (swap_xy):
            self.Q_state = Q_state.toarray()
            self.R_input = R_input.toarray()
            self.Q_state[0,0], self.Q_state[1,1], self.Q_state[2,2], self.Q_state[3,3], = self.Q_state[1,1], self.Q_state[0,0], self.Q_state[3,3], self.Q_state[2,2]
            self.R_input[0,0], self.R_input[1,1] = self.R_input[1,1], self.R_input[0,0]
            self.Q_state = sparse.dia_array(self.Q_state)
            self.R_input = sparse.dia_array(self.R_input)
        self.R_slack = R_slack
        self.V_ecr = V_ecr
        self.Nx = horizons["Nx"]
        self.Nc = horizons["Nc"]
        self.Nb = horizons["Nb"]
        self.u_lim = u_lim


class FailsafeParams:
    """
    This class encapsulates the tunable parameters of the LQR failsafe (homing) and deadbeat collision avoidance controllers utilized in simulation
    """
    def __init__(self, Q_fail, R_fail, C_int, K_dead):
        """
        Constructor for the FailsafeParams class
        Args:
            Q_fail: State penalty matrix for LQR failsafe (homing) controller
            R_fail: Input penalty matrix for LQR failsafe (homing) controller
            C_int: Output matrix that describes which output is used for integral state in the LQR controller
            K_dead: Gain matrix of the deadbeat collision avoidance controller
        """
        self.Q_fail = Q_fail
        self.R_fail = R_fail
        self.C_int = C_int
        self.K_dead = K_dead


def figurePlotSave(sim_conditions:SimConditions, debris:Debris, sim_run:SimRun, saveCounter=None):
    r"""
    This function plots a general simulation run. If a counter is provided, it catalogues and saves the plot in test\\RunFigs.

    :param sim_conditions: A SimConditions object representing general simulation conditions (initial state, orbital parameters, etc.)
    :param debris: A Debris object containing information describing the debris to be avoided by the control algorithm during simulation
    :param sim_run: A SimRun object with the necessary telemetry for plotting
    :param saveCounter: Counter for automatic saving of plots over multiple runs
    """

    # Unpack sim_run object
    xtruePiece = sim_run.x_true_pcw
    xestO = sim_run.x_est
    noiseStored = sim_run.noise_hist
    ctrls = sim_run.ctrl_hist
    iterm = sim_run.i_term
    controllerSeq = sim_run.ctrlr_seq

    # Helper function that determines the color of plotted trajectory based on controller used at each time step
    def numberToColor(num):
        if (num == 1 or num == 0):  # MPC controller
            col = 'b'
        elif (num == 2):    # LQR failsafe controller
            col = 'r'
        elif (num == 3):    # Deadbeat debris collision avoidance controller
            col = 'y'
        return col

    # Unpack simulation conditions
    gam = sim_conditions.los_ang
    rp = sim_conditions.r_p
    rtot = sim_conditions.r_tol
    phi = sim_conditions.hatch_ofst
    n = sim_conditions.mean_mtn
    T = sim_conditions.time_stp
    T_cont = sim_conditions.T_cont
    time_final = sim_conditions.T_final

    rx = sim_conditions.xr[0]
    ry = sim_conditions.xr[1]

    # Debris bounding box
    if debris is not None:
        sqVerts = debris.constructVertArr()

    # X values for LOS cone plot
    xInt = 0.1
    if (sim_conditions.inTrack):
        xSampsU = np.arange(-20, 0+xInt, xInt)
        xSampsL = np.arange(0, 20+xInt, xInt)
    else:
        xSampsU = np.arange(0, 110, xInt)
        xSampsL = xSampsU

    # Handle discrete and continuous time simulations
    if (math.isnan(T_cont)):
        uTime = [T * x for x in range(1, iterm + 1)]
        xTimeC = [T * x for x in range(iterm)]
        xTimeD = xTimeC
        itermD = iterm
    else:
        uTime = [T_cont * x for x in range(1, iterm + 1)]
        itermD = int(iterm/(T/T_cont))
        xTimeD = np.arange(0, time_final, T)[:itermD]
        xTimeC = np.arange(0, time_final, T_cont)[:iterm]
        #xTimeD = xTimeC

    # Construct velocity 1-norms    TODO replace redundant code
    xv1n = np.empty(xtruePiece.shape[1])
    for i in range(xtruePiece.shape[1]):
        xv1n[i] = np.absolute(xtruePiece[2, i]) + np.absolute(xtruePiece[3, i])

    # Plotting constraints and obstacles
    yVertSamps = np.arange(-10, 10 + xInt, xInt)
    xVertSamps = np.ones(yVertSamps.shape)
    yConeL = ((rp - rtot) * math.sin(gam) / (math.cos(phi - gam))) + math.tan(phi - gam) * xSampsL
    yConeU = -((rp - rtot) * math.sin(gam) / (math.cos(phi + gam))) + math.tan(phi + gam) * xSampsU
    vertCons = rp
    xVertSamps = xVertSamps * vertCons
    xCirc = np.arange(-rp, rp + xInt, xInt)
    xCircSq = np.square(xCirc)
    topCircle = np.sqrt(rp ** 2 - np.round(xCircSq, 2))
    botCircle = -np.sqrt(rp ** 2 - np.round(xCircSq, 2))

    # State-space trajectory and constraint plots
    if (sim_conditions.inTrack):
        ConsComb, (geoConp, velConp) = plt.subplots(nrows=1, ncols=2)
        ConsComb.set_size_inches((7, 5))
        ConsComb.set_dpi(300)
    else:
        ConsComb, (geoConp, velConp) = plt.subplots(nrows=2, ncols=1)
        ConsComb.set_size_inches((5,5.5))
        ConsComb.set_dpi(300)

    if debris is not None:
        geoConp.plot(np.array([sqVerts[1, 0], sqVerts[0, 0]]), np.array([sqVerts[1, 1], sqVerts[0, 1]]), color='#994F00', label='_nolegend_')
        geoConp.plot(np.array([sqVerts[1, 0], sqVerts[0, 0]]), np.array([sqVerts[1, 1], sqVerts[0, 1]]), color='#994F00', label='_nolegend_')
        geoConp.plot(np.array([sqVerts[2, 0], sqVerts[3, 0]]), np.array([sqVerts[2, 1], sqVerts[3, 1]]), color='#994F00', label='_nolegend_')
        geoConp.plot(np.array([sqVerts[2, 0], sqVerts[2, 0]]), np.array([sqVerts[2, 1], sqVerts[1, 1]]), color='#994F00', label='_nolegend_')
        geoConp.plot(np.array([sqVerts[3, 0], sqVerts[3, 0]]), np.array([sqVerts[3, 1], sqVerts[0, 1]]), color='#994F00', label='_nolegend_')

    geoConp.plot(xCirc, topCircle, color='0.5', label='_nolegend_')
    geoConp.plot(xCirc, botCircle, color='0.5', label='_nolegend_')
    geoConp.plot(xSampsL, yConeL, color='#994F00',label='Constraints')
    geoConp.plot(xSampsU, yConeU, color='#994F00', label='_nolegend_')
    if (sim_conditions.inTrack):
        geoConp.plot(yVertSamps, xVertSamps, color='#994F00', label='_nolegend_')
    else:
        geoConp.plot(xVertSamps, yVertSamps, color='#994F00', label='_nolegend_')
    for i in range(iterm - 1):
        geoConp.plot(xtruePiece[0, i:i + 2], xtruePiece[1, i:i + 2], color=numberToColor(controllerSeq[i + 1]), label='Trajectory')
    customLines = [Line2D([0], [0], color='b'),
                   Line2D([0], [0], color='r'),
                   Line2D([0], [0], color='y')]
    geoConp.set_aspect('equal')
    if (sim_conditions.inTrack):
        ConsComb.suptitle('Trajectory and Contraints (LVLH)')
    else:
        geoConp.title.set_text('Trajectory and Contraints (LVLH)')
    geoConp.set_ylabel('$\mathregular{\delta}$y (m)')
    geoConp.set_xlabel('$\mathregular{\delta}$x (m)')

    if (sim_conditions.noise is not None):
        geoConp.legend(customLines, ['MPC Controller', 'LQR Failsafe', 'LQR Debris Avoidance'], loc='lower right', prop={'size': 5})
    else:
        if (not sim_conditions.inTrack):
            geoConp.legend(['Constraints', 'Trajectory'], loc='lower left', prop={'size': 6.5})
        else:
            geoConp.legend(['Constraints', 'Trajectory'], loc='upper left', prop={'size': 6.5})

    velConp.set_xlabel('Relative Position L1 Norm (m)')
    velConp.set_ylabel('Relative Velocity L1 Norm (m/s)')
    velConp.plot(np.abs(xtruePiece[0, :iterm + 1] - rx) + np.abs(xtruePiece[1, :iterm + 1] - ry),
                 np.abs(xtruePiece[0, :iterm + 1] - rx) + np.abs(xtruePiece[1, :iterm + 1] - ry), color='#994F00',
                 label='_nolegend_')
    velConp.plot(np.abs(xtruePiece[0, :iterm + 1] - rx) + np.abs(xtruePiece[1, :iterm + 1] - ry),
                 np.reshape(xv1n[:iterm], iterm), color='b', label='Relative Velocity L1 Norm')

    # State trajectory plots
    if (sim_conditions.noise is None):
        estTrueStates = plt.figure(2)
        x1p = plt.subplot2grid((4, 3), (0, 0), rowspan=1, colspan=3)
        x2p = plt.subplot2grid((4, 3), (1, 0), rowspan=1, colspan=3)
        x3p = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
        x4p = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)

        x1p.plot(xTimeC, xtruePiece[0, :iterm])
        x2p.plot(xTimeC, xtruePiece[1, :iterm])
        x3p.plot(xTimeC, xtruePiece[2, :iterm])
        x4p.plot(xTimeC, xtruePiece[3, :iterm])

        estTrueStates.set_size_inches((7, 7.5))
        estTrueStates.set_dpi(300)

        x1p.title.set_text('True and Estimated States (LVLH)')
        x1p.set_ylabel('$\mathregular{\delta}$x (m)')
        x1p.xaxis.set_visible(False)
        x2p.set_ylabel('$\mathregular{\delta}$y (m)')
        x2p.xaxis.set_visible(False)
        x3p.set_ylabel('$\mathregular{\delta\dot{x}}$ (m/s)')
        x3p.xaxis.set_visible(False)
        x4p.set_ylabel('$\mathregular{\delta\dot{y}}$ (m/s)')
        x4p.set_xlabel('Time (s)')

        estTrueStates.align_labels()

    else:

        estTrueStates = plt.figure(2)
        x1p = plt.subplot2grid((6, 3), (0, 0), rowspan=1, colspan=3)
        x2p = plt.subplot2grid((6, 3), (1, 0), rowspan=1, colspan=3)
        x3p = plt.subplot2grid((6, 3), (2, 0), rowspan=1, colspan=3)
        x4p = plt.subplot2grid((6, 3), (3, 0), rowspan=1, colspan=3)
        d1p = plt.subplot2grid((6, 3), (4, 0), rowspan=1, colspan=3)
        d2p = plt.subplot2grid((6, 3), (5, 0), rowspan=1, colspan=3)

        x1p.plot(xTimeC, xtruePiece[0, :iterm + 1])
        x1p.plot(xTimeD, xestO[0, :itermD])
        x2p.plot(xTimeC, xtruePiece[1, :iterm + 1])
        x2p.plot(xTimeD, xestO[1, :itermD])
        x3p.plot(xTimeC, xtruePiece[2, :iterm + 1])
        x3p.plot(xTimeD, xestO[2, :itermD])
        x4p.plot(xTimeC, xtruePiece[3, :iterm + 1])
        x4p.plot(xTimeD, xestO[3, :itermD])
        d1p.plot(xTimeD, noiseStored[0, :itermD])
        d1p.plot(xTimeD, xestO[4, :itermD])
        d2p.plot(xTimeD, noiseStored[1, :itermD])
        d2p.plot(xTimeD, xestO[5, :itermD])

        estTrueStates.set_size_inches((7, 7.5))
        estTrueStates.set_dpi(300)

        x1p.title.set_text('True and Estimated States (LVLH)')
        x1p.set_ylabel('$\mathregular{\delta}$x (m)')
        x1p.legend(['Ground Truth','Estimated'], loc='upper right')
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

    # Control history plot
    controlPlot = plt.figure(3)
    u1p = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=3)
    u2p = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=3)

    u1p.plot(uTime, ctrls[0, :iterm])
    u2p.plot(uTime, ctrls[1, :iterm])

    if (not sim_conditions.isDeltaV):
        u1p.title.set_text('Actuator Commands (LVLH)')
        u1p.set_ylabel('$\mathregular{u_x}$ $\mathregular{(m/s^2)}$')
        u2p.set_ylabel('$\mathregular{u_y}$ $\mathregular{(m/s^2)}$')
    else:
        u1p.title.set_text('Actuator Commands (LVLH)')
        u1p.set_ylabel('$\mathregular{u_x}$ $\mathregular{(m/s)}$')
        u2p.set_ylabel('$\mathregular{u_y}$ $\mathregular{(m/s)}$')
    u2p.set_xlabel('Time (s)')

    # Label and save plots if provided a counter
    if saveCounter != None:
        iter = str(saveCounter) + '.png'
        direc = 'RunFigs/'
        estTrueStates.savefig(direc + 'trueANDest' + iter,dpi=300)
        controlPlot.savefig(direc + 'contrHist' + iter,dpi=300)
        ConsComb.savefig(direc + 'combCons' + iter,dpi=300)
        plt.close('all')
    else:
        plt.show()
        plt.close('all')


