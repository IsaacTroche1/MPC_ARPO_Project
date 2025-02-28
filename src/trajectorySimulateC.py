import osqp
import numpy as np
import scipy as sp
import sympy as sy
import math
from scipy import sparse
from numpy import random
import control as ct
from src.mpcsim import *
from src.simhelpers import *



def trajectorySimulateC(sim_conditions:SimConditions, mpc_params:MPCParams, fail_params:FailsafeParams, debris:Debris):

    # random.seed(123)

    isReject = sim_conditions.isReject
    inTrack = sim_conditions.inTrack
    distTol = sim_conditions.suc_cond[0]
    angTol = sim_conditions.suc_cond[1]
    noise = sim_conditions.noise

    #Noise characteristics
    if noise is not None:
        sigMat = sim_conditions.noise.constructSigMat()
        noiseRepeat = sim_conditions.noise.noise_length
    else:
        sigMat = np.diag([0.,0.,0.,0.])
        noiseRepeat = 1


    #Simulation Constants
    gam = sim_conditions.los_ang
    rp = sim_conditions.r_p
    rtot = sim_conditions.r_tol
    phi = sim_conditions.hatch_ofst
    n = sim_conditions.mean_mtn
    T = sim_conditions.time_stp

    #Contiuous time simulation setup
    T_cont = sim_conditions.T_cont
    time_final = sim_conditions.T_final
    nsimD = int(time_final / T)
    nsimC = int(time_final / T_cont)

    xTimeD = np.arange(0, time_final, T)
    xTimeC = np.arange(0, time_final, T_cont)

    def stateEqn(t, x, u, noise_vec):
        Ap = np.array([
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [3 * n ** 2, 0., 0., 2 * n],
            [0., 0., -2 * n, 0.],
        ])

        Bp = np.array([
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.],
        ])

        # u = np.array([[1],[1]])
        dxdt = Ap@x + Bp@u + noise_vec
        return dxdt

    # Initial and reference states
    x0 = sim_conditions.x0
    xr = sim_conditions.xr

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

    # Discrete time model of a quadcopter
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

    #Discretize
    Ad = sparse.csc_matrix(sp.linalg.expm(Ap*T))
    x = sy.symbols('x')
    Ax = sy.Matrix(Ap)*x
    eAx = Ax.exp()
    #eAxF = sy.lambdify((x),eAx)
    eAxInt = np.empty([Ap.shape[0],Ap.shape[0]])
    for (i,j), func_ij in np.ndenumerate(eAx):
        func = sy.lambdify((x),func_ij)
        eAxInt[i,j] = sp.integrate.quad(func,0.,T)[0]
    Bd = sparse.csc_matrix(eAxInt@Bp)

    #Observer and dynamic systems
    Ao = sp.linalg.block_diag(Ad.toarray(), Adi)
    Ao[0,4] = 1.
    Ao[1,5] = 1.
    Co = np.hstack([Cm, np.zeros([2,2])])

    # - input and state constraints
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


    # Constraints
    umin = np.hstack([-0.2, -0.2, np.zeros(ny)])
    umax = np.hstack([0.2, 0.2, np.inf*np.ones(ny)])
    Vecr = mpc_params.V_ecr


    D = np.hstack([np.zeros([ny,nu]),np.diag(Vecr)])

    # Objective function
    Q = mpc_params.Q_state
    Ru = mpc_params.R_input
    Rs = mpc_params.R_slack
    R = sparse.block_diag([Ru,Rs])

    #Virtual LQR Controller
    S = sp.linalg.solve_discrete_are(Ad.toarray(), Bd.toarray(), Q.toarray(), Ru.toarray())
    K = np.asarray(np.linalg.inv(Ru + np.transpose(Bd)@S@Bd)@(np.transpose(Bd)@S@Ad))
    QN = S

    #LQR Failsafe
    Qf = fail_params.Q_fail
    Rf = fail_params.R_fail
    Crefx = fail_params.C_int
    nr = Crefx.shape[0]

    Kf = ct.dlqr(Ad.toarray(),Bd.toarray(),Qf,Rf, integral_action=Crefx)[0]
    Kpf = Kf[:,:nx]
    Kif = Kf[:,nx:nx+nr]

    #Deadbeat Debris Avoidance
    Crefy = np.array([[0.,1.,0.,0.]])
    Bd_prune = np.reshape(Bd[:, 1], (nx, 1))[[1, 3],]
    Ad_prune = Ad[[1, 3], :][:, [1, 3]]
    C_prune = np.array([1, 0])

    A_aug = np.block([[Ad_prune.toarray(), np.zeros([2, 1])], [C_prune, np.eye(1)]])
    B_aug = np.block([[Bd_prune.toarray()], [np.zeros([1, 1])]])
    des_eig = np.array([0, 0, 0])
    K_prune = ct.acker(A_aug, B_aug, des_eig)

    act_eigs = np.linalg.eigvals(A_aug - B_aug @ K_prune)

    K_total = np.zeros([nu, nx])
    K_total[1, 1] = K_prune[0, 0]
    K_total[1, 3] = K_prune[0, 1]
    K_i = np.vstack([0, K_prune[0, 2]])


    if ~(np.all(np.linalg.eigvals(S) > 0)):
        raise Exception("Riccati solution not positive definite")


    # Horizons
    Nx = mpc_params.Nx
    Nc = mpc_params.Nc
    Nb = mpc_params.Nb

    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(Nx), Q), QN, sparse.kron(sparse.eye(Nc), R), 1*sparse.eye(ndi)], format='csc')
    # - linear objective
    q = np.hstack([np.kron(np.ones(Nx), -Q@xr), -QN@xr, np.zeros(Nc*(nu+ny)), np.zeros(ndi)])
    # - linear dynamics
    Aeq = constructOsqpAeq(mpc_params, Ad, Bd, K, ny)
    leq = np.hstack([-x0, np.zeros(Nx*nx)])
    ueq = leq


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

    #Intial conditions
    xtrue0 = x0
    xest0 = np.hstack([x0, 0., 0.]) #note this assumes we dont know the inital distrubance value (zeros)
    Pest = 10*np.eye(nx+ndi)

    # Simulate in closed loop
    iterm = nsimC
    ifailsd = []
    ifailsf = []
    impc = []
    xtrueP = np.empty([nx,nsimC])
    xestO = np.empty([nx+ndi,nsimD+1])
    xintf = 0
    noiseStored = np.empty([nx,nsimC])
    ctrls = np.empty([nu,nsimC])
    ctrls[:,0] = np.array([0.,0.])
    xv1n = np.empty([1,nsimC])
    xv1n[0,:int(T/T_cont)+1] = np.kron(np.ones([1,int(T/T_cont)+1]), (xtrueP[2,0]+xtrueP[3,0]))
    xtrueP[:,:int(T/T_cont)+1] = np.kron(np.ones([1,int(T/T_cont)+1]),xtrue0.reshape(-1,1))
    xestO[:,0] = xest0
    noiseVec = sigMat@random.normal(0, 1, 4)
    noiseStored[:,0] = noiseVec

    Bou = np.vstack([Bd.toarray(), np.zeros([2,2])])
    Bnoise = np.vstack([np.zeros([nx,ndi]), (T*noiseRepeat)*np.eye(ndi)]) #try with T*eye
    Qw = np.diag([40*sigMat[0,0]**2, 40*sigMat[1,1]**2])
    Qw = Bnoise@Qw@np.transpose(Bnoise)

    disc_j = 1
    time = T
    for i in range(500, nsimC-1):

        #Terminate sim conditions
        if (not inTrack and (np.linalg.norm(xtrueP[0:2,i]) < rp or xtrueP[0,i] < rp - rtot)):
            iterm = i
            break
        elif (inTrack and (np.linalg.norm(xtrueP[0:2,i]) < rp or xtrueP[1,i] < rp - rtot)):
            iterm = i
            break

        pass

        if ((disc_j < nsimD) and (xTimeC[i] == xTimeD[disc_j])):

            # Solve
            res = prob.solve()

            #Check solver status
            if res.info.status != 'solved':
                if (xestO[0,disc_j] - (center[0] + sideLength/2) < 0 and xestO[0,disc_j] - (center[0] - sideLength/2) > 0 and xestO[1,disc_j] < (center[1] + sideLength/2) and xestO[1,disc_j] > (center[1] - sideLength/2)):
                    ifailsd.append(i)
                    #break
                    xintf = xintf + Crefy@xestO[:4,disc_j] - (center[1] + sideLength/2) #theres a potential bug here with the sign of the control, test in animation
                    ctrl = -K_total@xestO[:4,disc_j] - K_i@xintf
                else:
                    ifailsf.append(i)
                    #break
                    xintf = xintf + Crefx@xestO[:4,disc_j] - xr[0]
                    ctrl = -Kpf@xestO[:4,disc_j] - Kif@xintf
            else:
                impc.append(i)
                xintf = 0
                ctrl = res.x[(Nx+1)*nx:(Nx+1)*nx+nu]

            #Scale input if excees max/min
            if (np.linalg.norm(ctrl) > umax[0]):
                ctrl[0] = ctrl[0]*(umax[0]/np.linalg.norm(ctrl))
                ctrl[1] = ctrl[1]*(umax[0]/np.linalg.norm(ctrl))

            ctrls[:,i+1] = ctrl

            if (disc_j % noiseRepeat == 0):
                noiseVec = sigMat@random.normal(0, 1, 4)
                noiseStored[:, i] = noiseVec
            else:
                noiseVec = noiseVec
                noiseStored[:, i] = noiseVec
        else:
            if (bool(impc) and impc[-1] == i - 1):
                impc.append(i)
            elif (bool(ifailsf) and ifailsf[-1] == i - 1):
                ifailsf.append(i)
            elif (bool(ifailsd) and ifailsd[-1] == i - 1):
                ifailsd.append(i)

            ctrl = ctrls[:,i]
            ctrls[:,i+1] = ctrl
            noiseVec = noiseStored[:, i-1]
            noiseStored[:, i] = noiseVec

        soln = sp.integrate.solve_ivp(stateEqn, (time, time + T_cont), xtrueP[:, i], args=(ctrls[:, i+1], noiseStored[:,i]))
        xtrueP[:,i+1] = soln.y[:,-1]
        xv1n[0,i+1] = np.absolute(xtrueP[2,i+1]) + np.absolute(xtrueP[3,i+1])

        if ((disc_j < nsimD) and (xTimeC[i] == xTimeD[disc_j])):
            # Measurement and state estimate
            if (noise is not None):
                xnom = Ao @ xestO[:, disc_j] + Bou @ ctrls[:, i+1]
                Pest = Ao @ Pest @ np.transpose(Ao) + Qw
                L = Pest @ np.transpose(Co) @ sp.linalg.inv(Co @ Pest @ np.transpose(Co))
                ymeas = Cm @ xtrueP[:, i+1]
                xestO[:, disc_j + 1] = xnom + L @ (ymeas - Co @ xnom)
                Pest = (np.eye(nx + ndi) - L @ Co) @ Pest
            else:
                xestO[:,disc_j] = np.hstack([xtrueP[:,i+1], [0., 0.]])

            # Update initial state
            # also need to change to estimated state
            l[:nx] = -xestO[:4, disc_j]
            u[:nx] = -xestO[:4, disc_j]
            prob.update(l=l, u=u)

            # Reconfigure velocity constraint
            A, lineq, uineq = configureDynamicConstraints(sim_conditions, mpc_params, debris, xestO[:, disc_j], block_mats, u_lim)
            l[(Nx + 1) * nx:] = lineq
            u[(Nx + 1) * nx:] = uineq
            prob.update(Ax=A.data, l=l, u=u)

            disc_j = disc_j + 1

        time = time + T_cont
        # print(time)

    #Construct piecewise trajectory
    xtruePiece = np.empty([nx,iterm])
    xtruePiece[:,:int(T/T_cont)+1] = xtrueP[:,:int(T/T_cont)+1]
    xtruePiece[:,impc] = xtrueP[:,impc]
    xtruePiece[:,ifailsf] = xtrueP[:,ifailsf]
    xtruePiece[:,ifailsd] = xtrueP[:,ifailsd]

    #contruct velocity one norms
    xv1n_test = np.empty(xtruePiece.shape[1])
    for i in range(xtruePiece.shape[1]):
        xv1n_test[i] = np.absolute(xtruePiece[2,i]) + np.absolute(xtruePiece[3,i])

    #Decide if successul trajectory
    succTraj = False
    for i in range(iterm-1,0,-1):
        dist = np.linalg.norm(xtruePiece[0:2,i] - xr[0:2])
        ang = abs(math.atan(xtruePiece[3,i]/xtruePiece[2,i]))*(180/np.pi)
        if (dist <= distTol and ang <= angTol):
            succTraj = True
            break


    controllerSeq = np.empty(iterm)
    controllerSeq[:int(T/T_cont)] = 0
    for i in impc:
        controllerSeq[i] = 1
    for i in ifailsf:
        controllerSeq[i] = 2
    for i in ifailsd:
        controllerSeq[i] = 3
    controllerSeq[-1] = controllerSeq[-2]

    sim_run = SimRun(iterm, succTraj, xtruePiece, xestO, ctrls, controllerSeq, noiseStored)
    return sim_run