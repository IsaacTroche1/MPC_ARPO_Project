import math
from vpython import *
import numpy as np

scene.userzoom = False
scene.userspin = False
scene.userPan = False
scene.autoscale = False

#Animation constants
distScale = 1
rE = 6371e+03
h = 500000
platWidth = 1.5
mu = 3.986e+14
n = 1.107e-03
Vplat_mag = (rE+h)*n

#starting camera position
camInc = 20
d = 100 #distance from camera to target
scene.camera.pos = vector(rE+h - d, 0, -platWidth/2)
#scene.center = vector(rE+h, 0, -platWidth/2)
scene.forward = (vector(rE+h, 0, -platWidth/2) - scene.camera.pos)/mag(vector(rE+h, 0, -platWidth/2) - scene.camera.pos)
scene.up = vector(0,0,1)

#Simulation Constants
gam = 10*(np.pi/180)
rp = 2.5
rtot = 1.5
phi = np.pi/180
n = 1.107e-3
T = 0.5

dt = 0.001

def acceleration(targ,earth):
    rrel = targ.pos - earth.pos
    rrel_hat = rrel/mag(rrel)
    amag = mu/(mag(rrel)**2)
    acelVec = -amag*rrel_hat
    return acelVec

Earth = sphere(pos = vector(0,0,0), radius = (rE), color = color.blue, velocity = vector(0,0,0), make_trail = False)
#target = cylinder(pos = vector(rE+h, 0, -platWidth/2), axis = vector(0, 0, platWidth), radius = rp, color = color.red, velocity = vector(0, Vplat_mag, 0), make_trail = True, trail_radius = 0.5, interval = 10)
target = cylinder(pos = vector(rE+h, 0, -platWidth/2), axis = vector(0, 0, platWidth), radius = rp, color = color.red, velocity = vector(0, Vplat_mag, 0), make_trail = True)


print("cool")

theta = 0
thetaRel = math.atan2(target.pos.y - scene.camera.pos.y, target.pos.x - scene.camera.pos.x)
time = 0
while (True):
    rate(5000)


    # print("d = ")
    # print(mag(scene.center - scene.camera.pos))

    target.acceleration = acceleration(target, Earth)
    target.velocity = target.velocity + target.acceleration*dt
    target.pos = target.pos + target.velocity*dt

    thetaTarg = math.atan2(target.pos.y,target.pos.x)
    thetaCam = math.atan2(scene.camera.pos.y,scene.camera.pos.x)
    thetaRel = math.atan2(target.pos.y - scene.camera.pos.y, target.pos.x - scene.camera.pos.x)
    camDist = mag(target.pos - scene.camera.pos)
    time = time + dt
    print("Relative angle = %f, Target anomaly = %f, time = %f, Camera distance = %f" % (thetaRel*(180/np.pi), thetaTarg*(180/np.pi), time, camDist))

    #update camera position
    # scene.camera.pos = vector(target.pos.x - (mag(target.pos - scene.camera.pos))*math.cos(theta), target.pos.y - (mag(target.pos - scene.camera.pos))*math.sin(theta), -platWidth/2)
    # scene.forward = (target.pos - scene.camera.pos)/mag(target.pos - scene.camera.pos)
    scene.camera.pos = vector(target.pos.x - d*math.cos(theta), target.pos.y - d*math.sin(theta), -platWidth/2)
    scene.forward = (target.pos - scene.camera.pos)/mag(target.pos - scene.camera.pos)
    #thetaRel = math.atan2(target.pos.y - scene.camera.pos.y, target.pos.x - scene.camera.pos.x)
    #print(mag(target.pos - scene.camera.pos))

    
