r"""
Simple script to generate animations from pickled sim runs in test\\RunObjs
"""

from src.mpcsim import *
from src.trajectorySimulate import trajectorySimulate
from src.animateTrajectory import animateTrajectory
from src.trajectorySimulateC import trajectorySimulateC
import pickle as pkl

#debris setup
center = (40.,0.)
side_length = 5.
detect_dist = 20

debris = Debris(center, side_length, detect_dist)

infile = open('RunObjs/Run6589.pkl','rb')
objs = pkl.load(infile)
obj1 = objs['simcond']
obj2 = objs['simrun']
infile.close()

# figurePlotSave(obj1, debris, obj2)

animateTrajectory(obj1, obj2, debris)