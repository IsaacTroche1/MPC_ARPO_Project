from MPCrendezKALMANdisturb import *
import matplotlib.pyplot as plt


#Simulation Constants
gam = 10*(np.pi/180)
rp = 2.5
rtot = 2.49
phi = 0*(np.pi/180)
n = 1.107e-3
T = 0.5

#Plotting Constraints and Obstacles
xInt = 0.1
xSamps = np.arange(0,110,xInt)
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

# ifail1, xTime1, xestO1, xtrueP1, noiseStored1 = disturbanceReject(50, 0, False)
# ifail2, xTime2, xestO2, xtrueP2, noiseStored2 = disturbanceReject(50, 1, False)

rx = rp
ry = 0
xr = np.array([rx,ry,0.,0.])

#monte carlo loop around this
MCnum = 200
errNon = np.empty(MCnum)
errComp = np.empty(MCnum)
for i in range(MCnum):
    print(i)
    ifail1, xTime1, xestO1, xtrueP1, noiseStored1 = disturbanceReject(50, 0, False) #50 is good
    ifail2, xTime2, xestO2, xtrueP2, noiseStored2 = disturbanceReject(50, 1, False)
    errDist1 = np.linalg.norm(xtrueP1[:,ifail1]-xr)
    errDist2 = np.linalg.norm(xtrueP2[:,ifail2]-xr)
    errNon[i] = errDist1
    errComp[i] = errDist2


avgDistNon = np.mean(errNon)
avgDistComp = np.mean(errComp)

print(ifail1)
print(ifail2)

plt.figure(1)
plt.plot(xCirc,topCircle)
plt.plot(xCirc,botCircle)
plt.plot(xSamps,yConeL)
plt.plot(xSamps,yConeU)
plt.plot(xVertSamps,yVertSamps)
plt.plot(xtrueP1[0,:ifail1+1],xtrueP1[1,:ifail1+1], label = "scale 0")
plt.plot(xtrueP2[0,:ifail2+1],xtrueP2[1,:ifail2+1], label = "scale 1")
plt.legend(loc = "lower right")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

