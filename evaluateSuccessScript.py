from debrisFullTraj import *
from animateTrajectory import *
from matplotlib.lines import Line2D

def figurePlotSave(iterm, xtruePiece, xestO, ctrls, noiseStored, controllerSeq, saveCounter = None):

    def numberToColor(num):
        if (num == 1):
            col = 'b'
        elif (num == 2):
            col = 'r'
        elif (num == 3):
            col = 'y'
        return col

    #Simulation Constants
    gam = 10*(np.pi/180)
    rp = 2.5
    rtot = 1.5
    phi = 0*(np.pi/180)
    n = 1.107e-3
    T = 0.5

    rx = rp
    ry = 0

    #Debris bounding box
    center = np.array([40,0])
    sideLength = 5
    sqVerts = np.array([[center[0]+sideLength/2, center[1]+sideLength/2],
                        [center[0]-sideLength/2, center[1]+sideLength/2],
                        [center[0]-sideLength/2, center[1]-sideLength/2],
                        [center[0]+sideLength/2, center[1]-sideLength/2]])

    xInt = 0.1
    xSamps = np.arange(0,110,xInt)
    xTime = [T*x for x in range(iterm)]

    #contruct velocity one norms
    xv1n = np.empty(xtruePiece.shape[1])
    for i in range(xtruePiece.shape[1]):
        xv1n[i] = np.absolute(xtruePiece[2,i]) + np.absolute(xtruePiece[3,i])
    
    #Plotting Constraints and Obstacles
    yVertSamps = np.arange(-10,10+xInt,xInt)
    xVertSamps = np.ones(yVertSamps.shape)
    yConeL = ((rp-rtot)*math.sin(gam)/(math.cos(phi-gam))) + math.tan(phi-gam)*xSamps
    yConeU = -((rp-rtot)*math.sin(gam)/(math.cos(phi+gam))) + math.tan(phi+gam)*xSamps
    vertCons = rp*math.sin(gam)
    vertCons = rp
    xVertSamps = xVertSamps*vertCons
    xCirc = np.arange(-rp,rp+xInt,xInt)
    xCircSq = np.square(xCirc)
    topCircle = np.sqrt(rp**2-np.round(xCircSq,2))
    botCircle = -np.sqrt(rp**2-np.round(xCircSq,2))

    #plot MPC portion
    # TWODtraj = plt.figure(1)
    # plt.plot(np.array([sqVerts[1,0],sqVerts[0,0]]),np.array([sqVerts[1,1],sqVerts[0,1]]))
    # plt.plot(np.array([sqVerts[2,0],sqVerts[3,0]]),np.array([sqVerts[2,1],sqVerts[3,1]]))
    # plt.plot(np.array([sqVerts[2,0],sqVerts[2,0]]),np.array([sqVerts[2,1],sqVerts[1,1]]))
    # plt.plot(np.array([sqVerts[3,0],sqVerts[3,0]]),np.array([sqVerts[3,1],sqVerts[0,1]]))
    # plt.plot(xCirc,topCircle)
    # plt.plot(xCirc,botCircle)
    # plt.plot(xSamps,yConeL)
    # plt.plot(xSamps,yConeU)
    # plt.plot(xVertSamps,yVertSamps)
    # for i in range(iterm-1):
    #     plt.plot(xtruePiece[0,i:i+2],xtruePiece[1,i:i+2], color = numberToColor(controllerSeq[i+1]))
    # ax = plt.gca()
    # ax.set_aspect('equal')

    #plt.figure(figsize = (16,9), dpi = 80)
    ConsComb, (geoConp,velConp) = plt.subplots(nrows=2,ncols=1)
    ConsComb.set_size_inches((5,6))
    ConsComb.set_dpi(200)
    geoConp.plot(np.array([sqVerts[1,0],sqVerts[0,0]]),np.array([sqVerts[1,1],sqVerts[0,1]]),color='#00FF00')
    geoConp.plot(np.array([sqVerts[1,0],sqVerts[0,0]]),np.array([sqVerts[1,1],sqVerts[0,1]]),color='#00FF00')
    geoConp.plot(np.array([sqVerts[2,0],sqVerts[3,0]]),np.array([sqVerts[2,1],sqVerts[3,1]]),color='#00FF00')
    geoConp.plot(np.array([sqVerts[2,0],sqVerts[2,0]]),np.array([sqVerts[2,1],sqVerts[1,1]]),color='#00FF00')
    geoConp.plot(np.array([sqVerts[3,0],sqVerts[3,0]]),np.array([sqVerts[3,1],sqVerts[0,1]]),color='#00FF00')
    geoConp.plot(xCirc,topCircle, color='0.5')
    geoConp.plot(xCirc,botCircle, color='0.5')
    geoConp.plot(xSamps,yConeL,color='#00FF00')
    geoConp.plot(xSamps,yConeU,color='#00FF00')
    geoConp.plot(xVertSamps,yVertSamps,color='#00FF00')
    for i in range(iterm-1):
        geoConp.plot(xtruePiece[0,i:i+2],xtruePiece[1,i:i+2], color = numberToColor(controllerSeq[i+1]))
    customLines = [Line2D([0],[0],color='b'),
                   Line2D([0],[0],color='r'),
                   Line2D([0],[0],color='y')]
    geoConp.set_aspect('equal')
    geoConp.title.set_text('Trajectory and Contraints')
    geoConp.set_ylabel('$\delta$y LVLH (m)')
    geoConp.set_xlabel('$\delta$x LVLH (m)')
    geoConp.legend(customLines, ['MPC Controller','LQR Failsafe','LQR Debris Avoidance'],loc='lower right', prop={'size':5})
    velConp.set_xlabel('Relative Position L1 Norm (m)')
    velConp.set_ylabel('Relative Position L1 Norm (m)')
    velConp.plot(np.abs(xtruePiece[0,:iterm+1]-rx)+np.abs(xtruePiece[1,:iterm+1]-ry),np.abs(xtruePiece[0,:iterm+1]-rx)+np.abs(xtruePiece[1,:iterm+1]-ry),color='#00FF00',label='_nolegend_')
    velConp.plot(np.abs(xtruePiece[0,:iterm+1]-rx)+np.abs(xtruePiece[1,:iterm+1]-ry),np.reshape(xv1n[:iterm], iterm), color = 'b',label='Relative Velocity L1 Norm')
    velConp.legend(['Relative Velocity L1 Norm (m/s)'],loc='upper left', prop={'size':5})

    estTrueStates = plt.figure(2)
    x1p = plt.subplot2grid((6, 3), (0, 0), rowspan=1, colspan=3)
    x2p = plt.subplot2grid((6, 3), (1, 0), rowspan=1, colspan=3)
    x3p = plt.subplot2grid((6, 3), (2, 0), rowspan=1, colspan=3)
    x4p = plt.subplot2grid((6, 3), (3, 0), rowspan=1, colspan=3)
    d1p = plt.subplot2grid((6, 3), (4, 0), rowspan=1, colspan=3)
    d2p = plt.subplot2grid((6, 3), (5, 0), rowspan=1, colspan=3)

    x1p.plot(xTime,xtruePiece[0,:iterm+1])
    x1p.plot(xTime,xestO[0,:iterm])
    x2p.plot(xTime,xtruePiece[1,:iterm+1])
    x2p.plot(xTime,xestO[1,:iterm])
    x3p.plot(xTime,xtruePiece[2,:iterm+1])
    x3p.plot(xTime,xestO[2,:iterm])
    x4p.plot(xTime,xtruePiece[3,:iterm+1])
    x4p.plot(xTime,xestO[3,:iterm])
    d1p.plot(xTime,noiseStored[0,:iterm])
    d1p.plot(xTime,xestO[4,:iterm])
    d2p.plot(xTime,noiseStored[1,:iterm])
    d2p.plot(xTime,xestO[5,:iterm])
    

    controlPlot = plt.figure(3)
    u1p = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=3)
    u2p = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=3)


    uTime = [T*x for x in range(1,iterm+1)]
    u1p.plot(uTime,ctrls[0,:iterm])
    u2p.plot(uTime,ctrls[1,:iterm])


    # velCon = plt.figure(4)
    # plt.plot(np.abs(xtruePiece[0,:iterm+1]-rx)+np.abs(xtruePiece[1,:iterm+1]-ry),np.abs(xtruePiece[0,:iterm+1]-rx)+np.abs(xtruePiece[1,:iterm+1]-ry))
    # plt.plot(np.abs(xtruePiece[0,:iterm+1]-rx)+np.abs(xtruePiece[1,:iterm+1]-ry),np.reshape(xv1n[:iterm], iterm))

    

    if saveCounter != None:
        iter = str(saveCounter) + '.png'
        direc = 'RunFigs/'
        # TWODtraj.savefig(direc + '2Dtraj' + iter)
        estTrueStates.savefig(direc + 'trueANDest' + iter)
        controlPlot.savefig(direc + 'contrHist' + iter)
        # velCon.savefig(direc + 'velConstraint' + iter)
        ConsComb.savefig(direc + 'combCons' + iter)
        plt.close('all')
    else:
        plt.show()
        plt.close('all')



# #while loop, if successful plot
# i = 0
# direc = 'RunArrays/'
# filename = 'Run'
# while (True):
#     iterm, success, xTruePiece, xestO, ctrls, controllerSeq, disturbs = debrisFullTraj(1, 50000, 50, 0.2, 45)
#     print(success)
#     if (success):
#         figurePlotSave(iterm, xTruePiece, xestO, ctrls, disturbs, controllerSeq, saveCounter=i)
#         np.savez(direc + filename + str(i), xTruePiece=xTruePiece, ctrls=ctrls, disturbs=disturbs, controllerSeq=controllerSeq)
#         #animateTrajectory(xTruePiece, ctrls, colorList=colorList, disturbs=disturbs)
#     i = i + 1

# cool = np.load(direc + filename + str(i) + '.npz')
# coolTrue = cool['xTruePiece']
# coolCtrls = cool['ctrls']
# coolDist = cool['disturbs']
# coolSeq = cool['controllerSeq']

iterm, success, xTruePiece, xestO, ctrls, controllerSeq, disturbs = debrisFullTraj(1, 50000, 50, 0.2, 45)
figurePlotSave(iterm, xTruePiece, xestO, ctrls, disturbs, controllerSeq)

#for loop, if successful add to successful percentage