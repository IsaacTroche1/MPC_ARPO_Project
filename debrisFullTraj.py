import osqp
import numpy as np
import scipy as sp
import sympy as sy
import math
from scipy import sparse
from numpy import random
import control as ct
from mpcsim import (SimConditions,MPCParams,Debris,FailsafeParams, SimRun)



def debrisFullTraj(sim_conditions:SimConditions, mpc_params:MPCParams, debris:Debris, fail_params:FailsafeParams):

    isReject = sim_conditions.isReject
    noiseRepeat = sim_conditions.noise.noise_length
    distTol = sim_conditions.suc_cond[0]
    angTol = sim_conditions.suc_cond[1]

    #Noise characteristics
    sigMat = sim_conditions.noise.constructSigMat()
    #random.seed(123)

    #Simulation Constants
    gam = sim_conditions.los_ang
    rp = sim_conditions.r_p
    rtot = sim_conditions.r_tol
    phi = sim_conditions.hatch_ofst
    n = sim_conditions.mean_mtn
    T = sim_conditions.time_stp

    # Initial and reference states
    x0 = sim_conditions.x0
    xr = sim_conditions.xr
    rx = xr[0]
    ry = xr[1]

    #Debris bounding box
    sqVerts = debris.constructVertArr()

    #delete these eventually
    center = debris.center
    sideLength = debris.side_length

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

    Bdi = 1*np.eye(2) #try with T instead of 1

    Cm = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    ])

    Cdi = np.eye(2)

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

    #LQR Stuff
    S = sp.linalg.solve_discrete_are(Ad.toarray(), Bd.toarray(), Q.toarray(), Ru.toarray())
    K = np.asarray(np.linalg.inv(Ru + np.transpose(Bd)@S@Bd)@(np.transpose(Bd)@S@Ad))
    QN = S

    #LQR Failsafe
    Qf = 0.005*np.diag([0.0001, 1, 100000., 1., 0.01])
    Rf = 100*np.diag([1, 1])
    Crefx = np.eye(1,nx)
    nr = Crefx.shape[0]

    Kf = ct.dlqr(Ad.toarray(),Bd.toarray(),Qf,Rf, integral_action=Crefx)[0]
    Kpf = Kf[:,:nx]
    Kif = Kf[:,nx:nx+nr]

    #LQR Debris Avoidance
    Qd = np.diag([0.00001, 50., 0.00001, 0.0001, 350.])  #for y
    Rd = 100000*np.diag([1., 1.])
    Crefy = np.array([[0,1,0,0]])
    nr = Crefy.shape[0]

    Kf = ct.dlqr(Ad.toarray(),Bd.toarray(),Qd,Rd, integral_action=Crefy)[0]
    Kpd = Kf[:,:nx]
    Kid = Kf[:,nx:nx+nr]


    if ~(np.all(np.linalg.eigvals(S) > 0)):
        raise Exception("Riccati solution not positive definite")


    # Horizons
    Nx = mpc_params.Nx
    Nc = mpc_params.Nc
    Nb = mpc_params.Nb

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(Nx), Q), QN,
                        sparse.kron(sparse.eye(Nc), R), 1*sparse.eye(ndi)], format='csc')
    # - linear objective
    q = np.hstack([np.kron(np.ones(Nx), -Q@xr), -QN@xr, np.zeros(Nc*(nu+ny)), np.zeros(ndi)])
    # - linear dynamics
    Ax1 = sparse.kron(sparse.eye(Nc+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(Nc+1, k=-1), Ad)
    Ax2 = sparse.kron(sparse.eye(Nx-Nc),-sparse.eye(nx)) + sparse.kron(sparse.eye(Nx-Nc, k=-1), (Ad-Bd@K))
    Ax3 = sparse.block_diag([Ax1, Ax2], format='csr')
    Ax4 = sparse.csr_matrix((Nx+1,Nx+1))
    Ax4[Nc+1,Nc] = 1
    Ax4 = sparse.kron(Ax4, (Ad-Bd@K))
    Ax = Ax3 + Ax4
    BuI = sparse.vstack([sparse.csc_matrix((1, Nc)), sparse.eye(Nc), sparse.csc_matrix((Nx-Nc, Nc))])
    Bdaug = sparse.hstack([Bd, np.zeros([nx,ny])])
    Bu = sparse.kron(BuI, Bdaug)
    Aeq = sparse.hstack([Ax, Bu])
    leq = np.hstack([-x0, np.zeros(Nx*nx)])
    ueq = leq


    Aineq1 = sparse.kron(sparse.eye(Nx+1), C)
    Aineq2 = sparse.kron(sparse.eye(Nc), sparse.eye(nu+ny))
    Block12 = sparse.vstack([np.kron(np.eye(Nc),D), np.kron(np.zeros([(Nx+1)-Nc,Nc]),np.zeros([ny,nu+ny]))])
    Block21 = sparse.coo_matrix((Nc*(nu+ny),(Nx+1)*nx))
    Aineq = sparse.block_array(([Aineq1, Block12],[Block21, Aineq2]), format='dia')
    xmin = np.array([1., 1., rp, 0.,-np.inf])
    xmax = np.array([np.inf, np.inf, np.inf, np.absolute(x0[0]-rx) + np.absolute(x0[1]-ry), np.inf]) 
    lineq = np.hstack([np.kron(np.ones(Nb+1), xmin), np.kron(np.ones(Nx-Nb),-np.inf*np.ones(ny)), np.kron(np.ones(Nc), umin), np.zeros([ndi,])]) #assume 0 est disturbance at start
    uineq = np.hstack([np.kron(np.ones(Nb+1), xmax), np.kron(np.ones(Nx-Nb), np.inf*np.ones(ny)), np.kron(np.ones(Nc), umax), np.zeros([ndi,])])
    # lineq[:(Nc)*ny] = lineq[:(Nc)*ny] - np.kron(np.ones([Nc,1]),np.diag(Vecr))@s0
    # uineq[:(Nc)*ny] = uineq[:(Nc)*ny] + np.kron(np.ones([Nc,1]),np.diag(Vecr))@s0
    # lineq = np.hstack([np.kron(np.ones(Nb+1), xmin), np.kron(np.ones(Nx-Nb),-np.inf*np.ones(ny)), np.kron(np.ones(Nc), umin)]) 
    # uineq = np.hstack([np.kron(np.ones(Nb+1), xmax), np.kron(np.ones(Nx-Nb), np.inf*np.ones(ny)), np.kron(np.ones(Nc), umax)])
    # - OSQP constraint
    A = sparse.vstack([Aeq, Aineq], format='csc')
    AextCol = sparse.vstack([np.zeros([nx,ndi]), np.kron(np.ones([Nx,1]),np.vstack([np.eye(ndi),np.zeros([nx-ndi,ndi])])), np.kron(np.zeros([(Nx+1),1]), np.zeros([ny,ndi])), np.kron(np.zeros([(Nc),1]), np.zeros([nu+ny,ndi]))])
    AextRow = sparse.csc_matrix(np.hstack([np.kron(np.ones([1,Nx+1]),np.zeros([ndi,nx])), np.kron(np.ones([1,Nc]),np.zeros([ndi,nu+ny])), np.eye(ndi)]))
    A = sparse.hstack([A, AextCol])
    A = sparse.vstack([A, AextRow])
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    # Create an OSQP object
    prob = osqp.OSQP()

    #Setup workspace
    prob.setup(P, q, A, l, u, warm_start=True, verbose = False)

    #Intial conditions
    xtrue0 = x0
    xest0 = np.hstack([x0, 0., 0.]) #note this assumes we dont know the inital distrubance value (zeros)
    Pest = 10*np.eye(nx+ndi)

    # Simulate in closed loop
    nsim = 1300
    iterm = nsim
    ifailsd = []
    ifailsf = []
    impc = []
    xtrueP = np.empty([nx,nsim+1])
    xestO = np.empty([nx+ndi,nsim+1])
    xintf = 0
    noiseStored = np.empty([nx,nsim+1])
    ctrls = np.empty([nu,nsim])
    xv1n = np.empty([1,nsim+1])
    xv1n[0,0] = xtrueP[2,0] + xtrueP[3,0]
    xtrueP[:,0] = xtrue0
    xestO[:,0] = xest0  
    noiseVec = sigMat@random.normal(0, 1, 4)
    noiseStored[:,0] = noiseVec

    Bou = np.vstack([Bd.toarray(), np.zeros([2,2])])
    Bnoise = np.vstack([np.zeros([nx,ndi]), (T*noiseRepeat)*np.eye(ndi)]) #try with T*eye
    Qw = np.diag([40*sigMat[0,0]**2, 40*sigMat[1,1]**2])
    Qw = Bnoise@Qw@np.transpose(Bnoise)
    for i in range(nsim):

        if (np.linalg.norm(xtrueP[0:2,i]) < rp or xtrueP[0,i] < rp - rtot):
            iterm = i
            break

        # Solve
        res = prob.solve()

        #Check solver status
        if res.info.status != 'solved':
            if (xestO[0,i] - (center[0] + sideLength/2) < 0 and xestO[0,i] - (center[0] - sideLength/2) > 0 and xestO[1,i] < (center[1] + sideLength/2) and xestO[1,i] > (center[1] - sideLength/2)):
                ifailsd.append(i)
                #break
                xintf = xintf + Crefy@xtrueP[:,i] - (center[1] + sideLength/2)
                ctrl = -Kpd@xestO[:4,i] - Kid@xintf
            else:
                ifailsf.append(i)
                #break
                xintf = xintf + Crefx@xtrueP[:,i] - xr[0]
                ctrl = -Kpf@xestO[:4,i] - Kif@xintf
        else:
            impc.append(i)
            xintf = 0
            ctrl = res.x[(Nx+1)*nx:(Nx+1)*nx+nu]

        #Scale input if excees max/min
        if (np.linalg.norm(ctrl) > umax[0]):
            ctrl[0] = ctrl[0]*(umax[0]/np.linalg.norm(ctrl))
            ctrl[1] = ctrl[1]*(umax[0]/np.linalg.norm(ctrl))


        # Apply first control input to the plant
        ctrls[:,i] = ctrl
        #remmembr to change to estimated state
        xtrueP[:,i+1] = Ad@xtrueP[:,i] + Bd@ctrl + noiseVec
        xv1n[0,i+1] = np.absolute(xtrueP[2,i+1]) + np.absolute(xtrueP[3,i+1])


        #Measurement and state estimate
        xnom = Ao@xestO[:,i] + Bou@ctrl
        Pest = Ao@Pest@np.transpose(Ao) + Qw
        L = Pest@np.transpose(Co)@sp.linalg.inv(Co@Pest@np.transpose(Co))
        ymeas = Cm@xtrueP[:,i]
        xestO[:,i+1] = xnom + L@(ymeas - Co@xnom)
        Pest = (np.eye(nx+ndi) - L@Co)@Pest

        # Update initial state
        #also need to change to estimated state
        l[:nx] = -xestO[:4,i+1]
        u[:nx] = -xestO[:4,i+1]
        prob.update(l=l, u=u)

        #Reconfigure velocity constraint
        C1 = (-1, 1)[xestO[2,i+1] >= 0]
        C2 = (-1, 1)[xestO[3,i+1] >= 0]
        if (xestO[0,i+1] - (center[0] + sideLength/2) < 0 and xestO[0,i+1] - (center[0] - sideLength/2) > 0):
            slope = (xestO[1,i+1]-sqVerts[1,1])/(xestO[0,i+1]-sqVerts[1,0])
            inter = -slope*xestO[0,i+1] + xestO[1,i+1]
        else:
            slope = (xestO[1,i+1]-sqVerts[0,1])/(xestO[0,i+1]-sqVerts[0,0])
            inter = -slope*xestO[0,i+1] + xestO[1,i+1]
        C = np.array([
                [C_11, C_12, 0., 0.],
                [C_21, C_22, 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., C1, C2],
                [-slope, 1., 0., 0.],
                ])
        Aineq1 = sparse.kron(sparse.eye(Nx+1), C)
        Aineq2 = sparse.kron(sparse.eye(Nc), sparse.eye(nu+ny))
        Block12 = sparse.vstack([np.kron(np.eye(Nc),D), np.kron(np.zeros([(Nx+1)-Nc,Nc]),np.zeros([ny,nu+ny]))])
        Block21 = sparse.coo_matrix((Nc*(nu+ny),(Nx+1)*nx))
        Aineq = sparse.block_array(([Aineq1, Block12],[Block21, Aineq2]), format='dia')
        A = sparse.vstack([Aeq, Aineq], format='csc')
        AextCol = sparse.vstack([np.zeros([nx,ndi]), np.kron(np.ones([Nx,1]),np.vstack([np.eye(ndi),np.zeros([nx-ndi,ndi])])), np.kron(np.zeros([(Nx+1),1]), np.zeros([ny,ndi])), np.kron(np.zeros([(Nc),1]), np.zeros([nu+ny,ndi]))])
        AextRow = sparse.csc_matrix(np.hstack([np.kron(np.ones([1,Nx+1]),np.zeros([ndi,nx])), np.kron(np.ones([1,Nc]),np.zeros([ndi,nu+ny])), np.eye(ndi)]))
        A = sparse.hstack([A, AextCol])
        A = sparse.vstack([A, AextRow])
        if (xestO[0,i+1] - (center[0] + sideLength/2) < 0 and xestO[0,i+1] - (center[0] - sideLength/2) > 0):
            xmin = np.array([1., 1., rp, 0., inter])
        elif (xestO[0,i+1] - (center[0] + sideLength/2) < 20 and xestO[0,i+1] - (center[0] + sideLength/2) > 0):
            xmin = np.array([1., 1., rp, 0., inter])
        else:
            xmin = np.array([1., 1., rp, 0., -np.inf])
        xmax = np.array([np.inf, np.inf, np.inf, np.absolute(x0[0]-rx) + np.absolute(x0[1]-ry), np.inf])
        lineq = np.hstack([np.kron(np.ones(Nb+1), xmin), np.kron(np.ones(Nx-Nb),-np.inf*np.ones(ny)), np.kron(np.ones(Nc), umin), isReject*xestO[4:6,i+1]]) #assume 0 est disturbance at start
        uineq = np.hstack([np.kron(np.ones(Nb+1), xmax), np.kron(np.ones(Nx-Nb), np.inf*np.ones(ny)), np.kron(np.ones(Nc), umax), isReject*xestO[4:6,i+1]])
        l[(Nx+1)*nx:] = lineq
        u[(Nx+1)*nx:] = uineq
        prob.update(Ax = A.data, l=l, u=u)

        #Inject noise into plant
        if (i % noiseRepeat == 0):
            noiseVec = sigMat@random.normal(0, 1, 4)
            noiseStored[:,i+1] = noiseVec
        else: 
            noiseVec = noiseVec
            noiseStored[:,i+1] = noiseVec

    #Construct piecewise trajectory
    xtruePiece = np.empty([nx,iterm])
    xtruePiece[:,impc] = xtrueP[:,impc]
    xtruePiece[:,ifailsf] = xtrueP[:,ifailsf]
    xtruePiece[:,ifailsd] = xtrueP[:,ifailsd]

    #contruct velocity one norms
    xv1n_test = np.empty(xtruePiece.shape[1])
    for i in range(xtruePiece.shape[1]):
        xv1n_test[i] = np.absolute(xtruePiece[2,i]) + np.absolute(xtruePiece[3,i])

    #Decide if successul trajectory
    minAng = np.inf
    minDist = np.inf
    succTraj = False
    for i in range(iterm-1,0,-1):
        dist = np.linalg.norm(xtruePiece[0:2,i] - xr[0:2])
        ang = abs(math.atan(xtruePiece[3,i]/xtruePiece[2,i]))*(180/np.pi)
        if (dist < minDist):
            minDist = dist
            minDisti = i
        if (ang < minAng):
            minAng = ang
            minAngi = i
        if (dist <= distTol and ang <= angTol):
            succTraj = True
            break



    colorList = [0 for x in range(iterm)]
    for i in impc:
        colorList[i] = 'b'
    for i in ifailsf:
        colorList[i] = 'r'
    for i in ifailsd:
        colorList[i] = 'y'

    controllerSeq = np.empty(iterm)
    for i in impc:
        controllerSeq[i] = 1
    for i in ifailsf:
        controllerSeq[i] = 2
    for i in ifailsd:
        controllerSeq[i] = 3

    xInt = 0.1
    xSamps = np.arange(0,110,xInt)
    xTime = [T*x for x in range(iterm)]

    sim_run = SimRun(iterm, succTraj, xtruePiece, xestO, ctrls, controllerSeq, noiseStored)
    return sim_run

#debrisFullTraj(1, 50000, 50, 0.2, 45)