import math
from vpython import *
import numpy as np
from src.mpcsim import *
import pickle


def animateTrajectory(sim_conditions:SimConditions, sim_run:SimRun, debris:Debris=None):

    xk = sim_run.x_true_pcw
    ctrls =  sim_run.ctrl_hist
    controllerSeq = sim_run.ctrlr_seq
    disturbs = sim_run.noise_hist #use noiseHist for saved runs 50 and 495

    def colorNumToObj(num):
        if (num == 1): #MPC
            colVec = vector(0,0,1)
        elif (num == 2): #LQR Failsafe
            colVec = vector(1,0,0)
        elif (num == 3): #Debris Avoid
            colVec = vector(1,1,0)
        return colVec

    nanim = xk.shape[1]

    if (controllerSeq.shape[0] == 0):
        controllerSeq = np.ones(nanim)

    scene = canvas(width = 900, height = 650, align = 'left')

    plot1 = graph(title='Control Inputs (ECI)', xtitle='Time (s)', ytitle='Control Input (N/kg)', width = 600, height = 300, align = 'right')
    if (disturbs.shape[0] != 0):
        plot2 = graph(title='Disturbances (ECI)', xtitle='Time (s)', ytitle='Positional Disturbance (m)', width = 600, height = 300, align = 'right')
        dxplot = gcurve(color = color.orange, label = 'X Disturbance', legend = True, graph = plot2)
        dyplot = gcurve(color = color.cyan, label = 'Y Distubance', legend = True, graph = plot2)

    uxplot = gcurve(color = color.blue, label = 'Ux', legend = True, graph = plot1)
    uyplot = gcurve(color = color.red, label = 'Uy', legend = True, graph = plot1)
    scene.select()

    scene.userzoom = True
    scene.userspin = False
    scene.userPan = False
    scene.autoscale = False

    #Animation constants
    rE = 6371e+03
    h = 500000
    platWidth = 1.5
    mu = 3.986e+14
    n = sim_conditions.mean_mtn
    Vplat_mag = (rE+h)*n

    #Debris
    if (debris is not None):
        center = debris.center
        sideLength = debris.side_length


    #Simulation Constants
    gam = sim_conditions.los_ang
    rp = sim_conditions.r_p
    rtot = sim_conditions.r_tol
    phi = sim_conditions.hatch_ofst
    n = sim_conditions.mean_mtn
    T = sim_conditions.time_stp

    #Animated constraints
    xInt = 0.1
    if (sim_conditions.inTrack):
        xSampsU = np.arange(-20, 0 + xInt, xInt)
        xSampsL = np.arange(0, 20 + xInt, xInt)
        first = 0
    else:
        xSampsU = np.arange(0, 110, xInt)
        xSampsL = xSampsU
        first = -1
    yConeL = ((rp-rtot)*math.sin(gam)/(math.cos(phi-gam))) + math.tan(phi-gam)*xSampsL
    yConeU = -((rp-rtot)*math.sin(gam)/(math.cos(phi+gam))) + math.tan(phi+gam)*xSampsU
    inputScale = 50
    disturbScale = 50

    dt = sim_conditions.time_stp

    def acceleration(targ,earth):
        rrel = targ.pos - earth.pos
        rrel_hat = rrel/mag(rrel)
        amag = mu/(mag(rrel)**2)
        acelVec = -amag*rrel_hat
        return acelVec

    Earth = sphere(pos = vector(0,0,0), radius = (rE), color = color.blue, velocity = vector(0,0,0), make_trail = False)
    Earth.visible = True
    target = cylinder(pos = vector(rE+h, 0, -platWidth/2), axis = vector(0, 0, 1), radius = rp, length = platWidth, color = color.gray(0.5), velocity = vector(0, Vplat_mag, 0), make_trail = True)
    yConeUpper = cylinder(pos = target.pos, axis = vector(-xSampsU[first],-yConeU[first],0), radius = 0.5, length = 100, color = vector(1,0.647,0.443), make_trail = False)
    yConeLower = cylinder(pos = target.pos, axis = vector(-xSampsL[-1],-yConeL[-1],0), radius = 0.5, length = 100, color = vector(1,0.647,0.443), make_trail = False)

    if (debris is not None):
        rDeb_eci = rotate(vector(center[0], center[1], 0), angle = np.pi, axis = vector(0,0,1))
        debris = box(pos = target.pos + rDeb_eci, axis = vector(0,0,1), size = vector(0.5,sideLength,sideLength), color = vector(1,0.647,0.443), make_trail = False)

    rChase0_eci = rotate(vector(xk[0,0], xk[1,0], 0), angle = np.pi, axis = vector(0,0,1))
    chaser = sphere(pos = rChase0_eci, radius = 0.5, color = color.purple, make_trail = False, trail_radius = 0.2)

    forceXc = vector(ctrls[0,0],0,0)
    forceYc = vector(0,ctrls[1,0],0)
    forceXeci = rotate(forceXc, angle = np.pi, axis = vector(0,0,1))
    forceYeci = rotate(forceYc, angle = np.pi, axis = vector(0,0,1))
    forceTotal_eci = forceXc + forceYeci
    inputX = arrow(pos = chaser.pos, axis = inputScale*vector(forceTotal_eci.x,0,0), color = colorNumToObj(controllerSeq[0]), shaftwidth = 0.5)
    inputY = arrow(pos = chaser.pos, axis = inputScale*vector(0,forceTotal_eci.y,0), color = colorNumToObj(controllerSeq[0]), shaftwidth = 0.5)

    if (disturbs.shape[0] != 0):
        distXc = vector(disturbs[0,0],0,0)
        distYc = vector(0,disturbs[1,0],0)
        distXeci = rotate(distXc, angle = np.pi, axis = vector(0,0,1))
        distYeci = rotate(distYc, angle = np.pi, axis = vector(0,0,1))
        distTotal_eci = distXeci + distYeci
        distX = arrow(pos = target.pos, axis = disturbScale*vector(distTotal_eci.x,0,0), color = color.green, shaftwidth = 0.5)
        distY = arrow(pos = target.pos, axis = disturbScale*vector(0,distTotal_eci.y,0), color = color.green, shaftwidth = 0.5)

    time = 0

    posOrth = vector(-target.pos.y, target.pos.x, 0)
    scene.camera.follow(chaser)
    scene.camera.rotate(5, posOrth, target.pos)
    if (sim_conditions.inTrack):
        scene.camera.rotate(90*(np.pi/180), vector(0,0,1), target.pos)
    scene.range = 30
    scene.up = vector(0,0,1)

    thetaTarg = 0
    thetaPlat = 0
    for i in range(1,nanim):
        rate(5)

        if (controllerSeq[i] == 1):
            scene.caption = '<b>Using controller: MPC</b>'
        elif (controllerSeq[i] == 2):
            scene.caption = '<b>Using controller: LQR Failsafe</b>'
        elif (controllerSeq[i] == 3):
            scene.caption = '<b>Using controller: Deadbeat Collision Avoidance</b>'
        
        # uxplot.plot(time, ctrls[0,i])
        # uyplot.plot(time, ctrls[1,i])
        uxplot.plot(time, forceTotal_eci.x)
        uyplot.plot(time, forceTotal_eci.y)

        if (disturbs.shape[0] != 0):
            dxplot.plot(time, distTotal_eci.x)
            dyplot.plot(time, distTotal_eci.y)


        target.rotate(angle = thetaPlat, axis = vector(0,0,1), origin = target.pos)
        yConeUpper.rotate(angle = thetaPlat, axis = vector(0,0,1), origin = target.pos)
        yConeLower.rotate(angle = thetaPlat, axis = vector(0,0,1), origin = target.pos)
        if (debris is not None):
            debris.rotate(angle = thetaPlat, axis = vector(0,0,1), origin = debris.pos)


        target.acceleration = acceleration(target, Earth)
        target.velocity = target.velocity + target.acceleration*dt
        target.pos = target.pos + target.velocity*dt

        yConeUpper.pos = target.pos
        yConeLower.pos = target.pos

        rotMat = np.array([[math.cos(thetaTarg + np.pi), -math.sin(thetaTarg + np.pi)],[math.sin(thetaTarg + np.pi),math.cos(thetaTarg + np.pi)]])
        if (debris is not None):
            rDeb_eci = rotMat@center
            debris.pos = target.pos + vector(rDeb_eci[0], rDeb_eci[1],0)

        rChase_eci = rotate(vector(xk[0,i], xk[1,i], 0), angle = np.pi + thetaTarg, axis = vector(0,0,1))
        chaser.pos = target.pos + rChase_eci
        chaser.make_trail = True

        forceXc = vector(ctrls[0,i],0,0)
        forceYc = vector(0,ctrls[1,i],0)
        forceXeci = rotate(forceXc, angle = np.pi + thetaTarg, axis = vector(0,0,1))
        forceYeci = rotate(forceYc, angle = np.pi + thetaTarg, axis = vector(0,0,1))
        inputX.pos = chaser.pos
        inputY.pos = chaser.pos
        forceTotal_eci = forceXeci + forceYeci
        inputX.axis = inputScale*vector(forceTotal_eci.x,0,0)
        inputY.axis = inputScale*vector(0,forceTotal_eci.y,0)
        inputX.color = colorNumToObj(controllerSeq[i])
        inputY.color = colorNumToObj(controllerSeq[i])
        
        if (disturbs.shape[0] != 0):
            distXc = vector(disturbs[0,i],0,0)
            distYc = vector(0,disturbs[1,i],0)
            distXeci = rotate(distXc, angle = np.pi + thetaTarg, axis = vector(0,0,1))
            distYeci = rotate(distYc, angle = np.pi + thetaTarg, axis = vector(0,0,1))
            distX.pos = target.pos
            distY.pos = target.pos
            distTotal_eci = distXeci + distYeci
            distX.axis = disturbScale*vector(distTotal_eci.x,0,0)
            distY.axis = disturbScale*vector(0,distTotal_eci.y,0)

        thetaTarg = math.atan2(target.pos.y,target.pos.x)
        thetaPlat = n*dt
        #print("Target anomaly = %f, time = %f" % (thetaTarg*(180/np.pi), time))
        
        scene.camera.rotate(thetaPlat, target.axis, target.pos)

        time = time + dt

#Testing area for debugging
# center = (40,0)
# side_length = 5
# debris = Debris(center,side_length)
#
# infile = open('saved_runs/Run495.pkl','rb')
# test_run = pickle.load(infile)
# infile.close()
#
# animateTrajectory(test_run, debris)