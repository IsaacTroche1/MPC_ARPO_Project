"""
This function performs trajectory simulation using an MPC controller on the CONTINUOUS-TIME, NONLINEAR
equations of relative motion.

Further documentation describing these objects can be found in the module src.mpcsim

The function utilizes a small selection of helpers to aid readability/conciseness.
These can be found in the module src.simhelpers
"""

import sympy as sy
import osqp
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from src.simhelpers import *

def trajectorySimulateC(sim_conditions:SimConditions, mpc_params:MPCParams, fail_params:FailsafeParams, debris:Debris):
    """
    Performs continuous-time nonlinear simulation

    :param sim_conditions: A SimConditions object representing general simulation conditions (initial state, orbital parameters, etc.)
    :param mpc_params: An MPCParams object containing the tunable parameters of the MPC controller to be used during simulation
    :param fail_params: A FailsafeParams object containing the tunable parameters of the failsafe controllers to be during simulation
    :param debris: A Debris object containing information describing the debris to be avoided by the control algorithm during simulation
    :return: A SimRun object with the necessary telemetry for data reduction and visualization/animation
    """

    # random.seed(123) # Uncomment for repeatable behavior

    isReject = sim_conditions.isReject
    inTrack = sim_conditions.inTrack
    isDeltaV = sim_conditions.isDeltaV
    distTol = sim_conditions.suc_cond[0]
    angTol = sim_conditions.suc_cond[1]
    noise = sim_conditions.noise

    # Noise characteristics
    if noise is not None:
        sigMat = sim_conditions.noise.constructSigMat()
        noiseRepeat = sim_conditions.noise.noise_length
    else:
        sigMat = np.diag([0.,0.,0.,0.])
        noiseRepeat = 1


    # Simulation Constants
    gam = sim_conditions.los_ang
    rp = sim_conditions.r_p
    rtot = sim_conditions.r_tol
    phi = sim_conditions.hatch_ofst
    n = sim_conditions.mean_mtn
    T = sim_conditions.time_stp

    # Continuous-time simulation setup
    T_cont = sim_conditions.T_cont
    time_final = sim_conditions.T_final
    nsimD = int(time_final / T)
    nsimC = int(time_final / T_cont)

    xTimeD = np.arange(0, time_final, T)
    xTimeC = np.arange(0, time_final, T_cont)

    # Function containing state equation for nonlinear plant, to be integrated with RK45
    def stateEqnN(t, x, u):

        # Orbital calcs for nonlinear plant (ASSUMES 500KM ALT. ORBIT)
        h = 500e+03
        re = 6378.1e+03
        R_T = h + re
        mu = (n ** 2) * (R_T ** 3)

        dxdt = [None] * 4

        dxdt[0] = x[2]
        dxdt[1] = x[3]
        dxdt[2] = 2 * n * x[3] + (n ** 2) * x[0] - (mu * (R_T + x[0])) / (((R_T + x[0]) ** 2 + x[1] ** 2) ** (3 / 2)) + mu / (R_T ** 2) + u[0]
        dxdt[3] = -2 * n * x[2] + (n ** 2) * x[1] - (mu * x[1]) / (((R_T + x[0]) ** 2 + x[1] ** 2) ** (3 / 2)) + u[1]

        return dxdt

    # Initial and reference states
    x0 = sim_conditions.x0
    xr = sim_conditions.xr

    # Debris setup
    if debris is not None:
        #Debris bounding box
        sqVerts = debris.constructVertArr()

        #delete these eventually
        center = debris.center
        sideLength = debris.side_length
        hasDebris = True
    else:
        center = (-np.inf,-np.inf)
        sideLength = 0
        hasDebris = False

    # Continuous-time linear CW equations
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
    [nx, nu] = Bp.shape

    Cm = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    ])

    [nx, nu] = Bp.shape
    nym = Cm.shape[0]

    # Discretize
    Ad = sparse.csc_matrix(sp.linalg.expm(Ap*T))
    x = sy.symbols('x')
    Ax = sy.Matrix(Ap)*x
    eAx = Ax.exp()
    eAxInt = np.empty([Ap.shape[0],Ap.shape[0]])
    for (i,j), func_ij in np.ndenumerate(eAx):
        func = sy.lambdify((x),func_ij)
        eAxInt[i,j] = sp.integrate.quad(func,0.,T)[0]
    if (not isDeltaV):
        Bd = sparse.csc_matrix(eAxInt @ Bp)
    else:
        Bd = sparse.csc_matrix(Ad @ np.vstack([np.zeros([2, 2]), np.eye(2)]))

    # State observer system for disturbance estimation
    Ao = sp.linalg.block_diag(Ad.toarray(), Adi)
    Ao[0,4] = 1.
    Ao[1,5] = 1.
    Bou = np.vstack([Bd.toarray(), np.zeros([2, 2])])
    Co = np.hstack([Cm, np.zeros([2,2])])

    # Measurement and state transition model for unscented kalman filter
    def fx(x, u):
        return Ao @ x + Bou @ u

    def hx(x):
        ymeas = np.empty(nym)
        ymeas[0] = np.linalg.norm(x[:2])
        ymeas[1] = math.atan2(x[1], x[0])
        return ymeas

    sig_points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2., kappa=-1)

    # - Define initial state constraints
    C_11 = math.sin(phi+gam)/((rp-rtot)*math.sin(gam))
    C_12 = -math.cos(phi+gam)/((rp-rtot)*math.sin(gam))
    C_21 = -math.sin(phi-gam)/((rp-rtot)*math.sin(gam))
    C_22 = math.cos(phi-gam)/((rp-rtot)*math.sin(gam))
    if (x0[0] - (center[0] + sideLength / 2) < 0 and x0[0] - (center[0] - sideLength / 2) > 0):
        slope = (x0[1] - sqVerts[1, 1]) / (x0[0] - sqVerts[1, 0])
        inter = -slope * x0[0] + x0[1]
    elif (hasDebris):
        slope = (x0[1] - sqVerts[0, 1]) / (x0[0] - sqVerts[0, 0])
        inter = -slope * x0[0] + x0[1]
    else:
        slope = 0
    C = np.array([
            [C_11, C_12, 0., 0.],
            [C_21, C_22, 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 1.],
            [-slope, 1., 0., 0.],
                ])

    if (inTrack):
        C[2,:] = np.array([0.,1.,0.,0.])

    ny = C.shape[0]


    # Define input constraints
    ulim = mpc_params.u_lim
    umin = np.hstack([-ulim[0], -ulim[1], np.zeros(ny)])
    umax = np.hstack([ulim[0], ulim[1], np.inf * np.ones(ny)])
    Vecr = mpc_params.V_ecr


    D = np.hstack([np.zeros([ny,nu]),np.diag(Vecr)])

    # State and input penalty matrices for MPC formulation
    Q = mpc_params.Q_state
    Ru = mpc_params.R_input
    Rs = mpc_params.R_slack
    R = sparse.block_diag([Ru,Rs])

    # Virtual LQR Controller for stabilization
    S = sp.linalg.solve_discrete_are(Ad.toarray(), Bd.toarray(), Q.toarray(), Ru.toarray())
    K = np.asarray(np.linalg.inv(Ru + np.transpose(Bd)@S@Bd)@(np.transpose(Bd)@S@Ad))
    QN = S

    # LQR Failsafe (homing controller)
    Qf = fail_params.Q_fail
    Rf = fail_params.R_fail
    Crefx = fail_params.C_int
    nr = Crefx.shape[0]

    Kf = ct.dlqr(Ad.toarray(),Bd.toarray(),Qf,Rf, integral_action=Crefx)[0]
    Kpf = Kf[:,:nx]
    Kif = Kf[:,nx:nx+nr]

    # Deadbeat debris avoidance maneuver
    Crefy = np.array([[0.,1.,0.,0.]])
    Bd_prune = np.reshape(Bd[:, 1], (nx, 1))[[1, 3],]
    Ad_prune = Ad[[1, 3], :][:, [1, 3]]
    C_prune = np.array([1, 0])

    A_aug = np.block([[Ad_prune.toarray(), np.zeros([2, 1])], [C_prune, np.eye(1)]])
    B_aug = np.block([[Bd_prune.toarray()], [np.zeros([1, 1])]])
    des_eig = np.array([0, 0, 0])
    K_prune = ct.acker(A_aug, B_aug, des_eig)

    K_total = np.zeros([nu, nx])
    K_total[1, 1] = K_prune[0, 0]
    K_total[1, 3] = K_prune[0, 1]
    K_i = np.vstack([0, K_prune[0, 2]])

    # For debugging
    if ~(np.all(np.linalg.eigvals(S) > 0)):
        raise Exception("Riccati solution not positive definite")


    # MPC horizons
    Nx = mpc_params.Nx
    Nc = mpc_params.Nc
    Nb = mpc_params.Nb

    # Quadratic objective for QP problem
    P = sparse.block_diag([sparse.kron(sparse.eye(Nx), Q), QN, sparse.kron(sparse.eye(Nc), R), 1*sparse.eye(ndi)], format='csc')
    # Linear objective for QP problem
    q = np.hstack([np.kron(np.ones(Nx), -Q@xr), -QN@xr, np.zeros(Nc*(nu+ny)), np.zeros(ndi)])
    # Linear dynamics constraint for QP problem
    Aeq = constructOsqpAeq(mpc_params, Ad, Bd, K, ny)
    leq = np.hstack([-x0, np.zeros(Nx*nx)])
    ueq = leq

    # Initialize state and input constraints for QP problem
    Aineq2 = sparse.kron(sparse.eye(Nc), sparse.eye(nu+ny))
    Block12 = sparse.vstack([np.kron(np.eye(Nc),D), np.kron(np.zeros([(Nx+1)-Nc,Nc]),np.zeros([ny,nu+ny]))])
    Block21 = sparse.coo_matrix((Nc*(nu+ny),(Nx+1)*nx))
    AextCol = sparse.vstack([np.zeros([nx, ndi]), np.kron(np.ones([Nx, 1]), np.vstack([np.eye(ndi), np.zeros([nx - ndi, ndi])])),np.kron(np.zeros([(Nx + 1), 1]), np.zeros([ny, ndi])), np.kron(np.zeros([(Nc), 1]), np.zeros([nu + ny, ndi]))])
    AextRow = sparse.csc_matrix(np.hstack([np.kron(np.ones([1, Nx + 1]), np.zeros([ndi, nx])), np.kron(np.ones([1, Nc]), np.zeros([ndi, nu + ny])),np.eye(ndi)]))

    block_mats = (Aeq, Aineq2, Block12, Block21, AextRow, AextCol, C)
    u_lim = (umin, umax)

    A, lineq, uineq = configureDynamicConstraints(sim_conditions, mpc_params, debris, np.hstack([np.copy(x0),0,0]), block_mats, u_lim)
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    block_mats = (Aeq,Aineq2,Block12,Block21,AextRow,AextCol,C)
    u_lim = (umin,umax)

    # Create an OSQP object
    prob = osqp.OSQP()

    #Setup workspace
    prob.setup(P, q, A, l, u, warm_start=True, verbose = False)

    # Initial conditions for simulation telemetry
    xtrue0 = x0
    xest0 = np.hstack([x0, 0., 0.]) #note this assumes we dont know the inital distrubance value (zeros)
    Pest = sp.linalg.block_diag(1e-20*np.eye(nx),np.eye(ndi))

    # This is just a mess :( TODO refactor
    iterm = nsimC
    ifailsd = []
    ifailsf = []
    impc = []
    xtrueP = np.empty([nx,nsimC])
    xestO = np.empty([nx+ndi,nsimD+1])
    xintf = 0
    noiseStored = np.empty([nx,nsimC])
    ctrls = np.empty([nu,nsimC])
    ctrls[:,:int(T/T_cont)+1] = np.kron(np.ones([1,int(T/T_cont)+1]),np.zeros([2,1]))
    xv1n = np.empty([1,nsimC])
    xv1n[0,:int(T/T_cont)+1] = np.kron(np.ones([1,int(T/T_cont)+1]), (xtrueP[2,0]+xtrueP[3,0]))
    xtrueP[:,:int(T/T_cont)+1] = np.kron(np.ones([1,int(T/T_cont)+1]),xtrue0.reshape(-1,1))
    xestO[:,0] = xest0

    # Continuous-time noise covariance setup
    Qcont = np.diag([sigMat[0,0]**2, sigMat[0,0]**2])
    noiseInterval = T*noiseRepeat
    noiseTimes = np.arange(0, time_final, noiseInterval)
    noiseIntC = int((noiseRepeat * T) / T_cont)
    noiseIntD = int((noiseRepeat) / T)
    V = ct.white_noise(noiseTimes, Qcont, dt=0.001)
    sum_vec = np.empty([nx,nsimD])
    j = 0
    for col in V.T:
        noiseStored[:, j * noiseIntC:noiseIntC * (1 + j)] = np.vstack([col.reshape(ndi, 1),np.zeros([2,1])])
        sum_vec[:, j * noiseRepeat:noiseRepeat * (1 + j)] = int(T/T_cont)*np.concatenate([col,np.zeros(2)]).reshape(-1,1)
        j += 1


    Bnoise = np.vstack([np.zeros([nx,ndi]), (T*int(T/T_cont))*np.eye(ndi)]) #try with T*eye
    Qw = np.diag([sigMat[0,0]**2, sigMat[1,1]**2])
    Qw = Bnoise@Qw@np.transpose(Bnoise)
    Qw[:4, :][:, :4] = 0.001 * np.eye(nx)

    # Setup UKF
    kf = UnscentedKalmanFilter(dim_x=6, dim_z=2, dt=T, fx=fx, hx=hx, points=sig_points)
    kf.x = xest0
    kf.P = Pest
    kf.R = np.zeros([nym, nym])
    kf.Q = Qw

    # Closed-loop simulation
    disc_j = 1
    time = T
    for i in range(500, nsimC-1):

        #Terminate simulation conditions
        if (not inTrack and (np.linalg.norm(xtrueP[0:2,i]) < rp or xtrueP[0,i] < rp - rtot)):
            iterm = i
            break
        elif (inTrack and (np.linalg.norm(xtrueP[0:2,i]) < rp or xtrueP[1,i] < rp - rtot)):
            iterm = i
            break

        if ((disc_j < nsimD) and (xTimeC[i] == xTimeD[disc_j])):    #Really crappy sample and hold

            # Attempt to solve QP problem
            res = prob.solve()

            # Check solver status to determine action
            if res.info.status != 'solved':
                if (xestO[0,disc_j-1] - (center[0] + sideLength/2) < 0 and xestO[0,disc_j-1] - (center[0] - sideLength/2) > 0 and xestO[1,disc_j-1] < (center[1] + sideLength/2) and xestO[1,disc_j-1] > (center[1] - sideLength/2)):
                    # Use deadbeat collision avoidance
                    ifailsd.append(i)
                    xintf = xintf + Crefy@xestO[:4,disc_j-1] - (center[1] + sideLength/2) #theres a potential bug here with the sign of the control, test in animation
                    ctrl = -K_total@xestO[:4,disc_j-1] - K_i@xintf
                else:
                    # Use failsafe (homing) controller
                    ifailsf.append(i)
                    xintf = xintf + Crefx@xestO[:4,disc_j-1] - xr[0]
                    ctrl = -Kpf@xestO[:4,disc_j-1] - Kif@xintf
            else:
                # Use MPC controller
                impc.append(i)
                xintf = 0
                ctrl = res.x[(Nx+1)*nx:(Nx+1)*nx+nu]

            #Scale input if desired exceeds max/min
            if (np.linalg.norm(ctrl) > umax[0]):
                ctrl[0] = ctrl[0]*(umax[0]/np.linalg.norm(ctrl))
                ctrl[1] = ctrl[1]*(umax[0]/np.linalg.norm(ctrl))

            ctrls[:,i+1] = ctrl

        else:
            continuousAppendIndex(impc, ifailsf, ifailsd, i)

            ctrl = ctrls[:,i]
            ctrls[:,i+1] = ctrl

        #Handle delta-v and acceleration input models
        if (not isDeltaV):
            soln = sp.integrate.solve_ivp(stateEqnN, (time, time + T_cont), xtrueP[:, i], args=(ctrls[:, i],))
            xtrueP[:,i+1] = soln.y[:,-1] + noiseStored[:,i]
        else:
            soln = sp.integrate.solve_ivp(stateEqnN, (time, time + T_cont), xtrueP[:, i], args=(np.zeros(nu),))
            if ((disc_j < nsimD) and (xTimeC[i] == xTimeD[disc_j])):
                xtrueP[:, i + 1] = soln.y[:, -1] + np.hstack([np.zeros(2), ctrls[:,i]]) + noiseStored[:, i]
            else:
                xtrueP[:, i + 1] = soln.y[:, -1] + noiseStored[:, i]

        xv1n[0,i+1] = np.absolute(xtrueP[2,i+1]) + np.absolute(xtrueP[3,i+1])

        if ((disc_j < nsimD) and (xTimeC[i] == xTimeD[disc_j])):    # Really crappy sample and hold
            # Measurement and state estimation
            if (noise is not None):
                ymeas = np.empty(nym)
                ymeas[0] = np.linalg.norm(xtrueP[:2, i+1])
                ymeas[1] = math.atan2(xtrueP[1, i+1], xtrueP[0, i+1])
                kf.predict(ctrls[:, i])
                kf.update(z=ymeas)
                xestO[:, disc_j] = kf.x
            else:
                xestO[:,disc_j] = np.hstack([xtrueP[:,i+1], [0., 0.]])

            # Update initial state in QP problem
            l[:nx] = -xestO[:4, disc_j]
            u[:nx] = -xestO[:4, disc_j]
            prob.update(l=l, u=u)

            # Reconfigure constraints
            A, lineq, uineq = configureDynamicConstraints(sim_conditions, mpc_params, debris, xestO[:, disc_j], block_mats, u_lim)
            l[(Nx + 1) * nx:] = lineq
            u[(Nx + 1) * nx:] = uineq
            prob.update(Ax=A.data, l=l, u=u)

            disc_j = disc_j + 1

        time = time + T_cont
        # print(time)
    continuousAppendIndex(impc, ifailsf, ifailsd, i + 1)

    # Construct piecewise trajectory (based on controller used)
    xtruePiece = np.empty([nx,iterm])
    xtruePiece[:,:int(T/T_cont)+1] = xtrueP[:,:int(T/T_cont)+1]
    xtruePiece[:,impc] = xtrueP[:,impc]
    xtruePiece[:,ifailsf] = xtrueP[:,ifailsf]
    xtruePiece[:,ifailsd] = xtrueP[:,ifailsd]

    # Contruct velocity 1-norms
    xv1n_test = np.empty(xtruePiece.shape[1])
    for i in range(xtruePiece.shape[1]):
        xv1n_test[i] = np.absolute(xtruePiece[2,i]) + np.absolute(xtruePiece[3,i])

    # Decide if trajectory was successful
    succTraj = False
    for i in range(iterm-1,0,-1):
        dist = np.linalg.norm(xtruePiece[0:2,i] - xr[0:2])
        ang = abs(math.atan(xtruePiece[3,i]/xtruePiece[2,i]))*(180/np.pi)
        if (dist <= distTol and ang <= angTol):
            succTraj = True
            break

    # Catalogue controller used at each point
    controllerSeq = np.empty(iterm)
    controllerSeq[:int(T/T_cont)] = 0
    for i in impc:
        controllerSeq[i] = 1    # A value of 1 means the MPC controller was used at this time step
    for i in ifailsf:
        controllerSeq[i] = 2    # A value of 2 means the LQR failsafe (homing) controller was used at this time step
    for i in ifailsd:
        controllerSeq[i] = 3    # A value of 3 means the deadbeat debris avoidance controller was used at this time step
    controllerSeq[-1] = controllerSeq[-2]

    sim_run = SimRun(iterm, succTraj, xtruePiece, xestO, ctrls, controllerSeq, sum_vec)
    return sim_run