r"""
This script is my playground for generating/tuning CONTINUOUS-TIME trajectories. It can definitely be structured better.

To generate trajectories continuously (as the process is fairly slow), I save the labeled figures in test\\RunFigs
and pickle the corresponding objects in test\\RunObjs.

Useful as a general template.
"""

from src.mpcsim import *
from src.trajectorySimulate import trajectorySimulate
from src.animateTrajectory import animateTrajectory
from src.trajectorySimulateC import trajectorySimulateC
from scipy import sparse
import pickle as pkl

# Debris definition
center = (40.,0.)
side_length = 5.
detect_dist = 20

# Noise definition
sig_x = 0.0012
sig_y = 0.0012
noise_length = 50

# Success conditions definition
distance_tolerance = 0.2
ang_tolerance = 45

# General sim conditions definition
platform_radius = 2.5
tolerance_radius = 1.5
los_angle = 10*(np.pi/180)
mean_motion = 1.107e-3
sample_time = 0.5
final_time = 300
cont_time_T = 0.001

in_track = False
x0 = np.array([100.,10.,0.,0])
rx = platform_radius
ry = 0
xr = np.array([rx,ry,0.,0.])

is_reject = True
is_deltav = False
success_cond = (distance_tolerance, ang_tolerance)
noises = Noise((sig_x,sig_y), noise_length)
# noises = None

# MPC controller parameters
Q_mpc = 8e+02*sparse.diags([0.2**2., 10**2., 3.8**2, 900]) #This is for radial approach, swap_xy in MPCparam init for auto in-track conversion
R_mpc = 1000**2*sparse.diags([1, 1])
R_mpc_s = 5**2*sparse.eye(5)  #make argument programmatic
ECRscale = 50000
v_ecr = ECRscale*np.ones(5) #0 for hard constraints
v_ecr[-2] = -1*v_ecr[-2]
v_ecr[-1] = 0
horizons = {"Nx":40,
            "Nc":5,
            "Nb":5}
ulim = (0.2, 0.2)

# LQR failsafe controller parameters
Q_failsafe = 0.005*np.diag([0.0001, 1, 100000., 1., 0.01])
R_failsafe = 100*np.diag([1, 1])
C_refx = np.eye(1,4)

# Populate parameter objects
sim_conditions = SimConditions(x0, xr, platform_radius, los_angle, tolerance_radius, mean_motion, sample_time, is_reject, success_cond, noises, in_track, T_cont=cont_time_T, T_final=final_time, isDeltaV=is_deltav)
mpc_params = MPCParams(Q_mpc, R_mpc, R_mpc_s, v_ecr, horizons, ulim)
debris = Debris(center, side_length, detect_dist)
# debris = None
fail_params = FailsafeParams(Q_failsafe,R_failsafe,C_refx,np.zeros([2,2]))




# Run simulations
sim_run_test = trajectorySimulateC(sim_conditions, mpc_params, fail_params, debris)
figurePlotSave(sim_conditions, debris, sim_run_test)
# outfile = open('RunObjs/test_run_cont_non.pkl','wb')
# pkl.dump({'simcond':sim_conditions,'simrun':sim_run_test,'debris':debris},outfile)
# outfile.close()
#
# infile = open('RunObjs/test_run_cont_non.pkl','rb')
# objs = pkl.load(infile)
# obj1 = objs['simcond']
# obj2 = objs['simrun']
# obj3 = objs['debris']
# infile.close()
#
# figurePlotSave(obj1, obj3, obj2)

# animateTrajectory(obj1, obj2, debris)

# Loop for generating and saving trajectories for later
# i = 0
# direc = 'RunObjs/'
# filename = 'Run'
# while (True):
#     sim_run_test = trajectorySimulateC(sim_conditions, mpc_params, fail_params, debris)
#     # figurePlotSave(sim_conditions, debris, sim_run_test)
#     print(sim_run_test.isSuccess)
#     if (sim_run_test.isSuccess):
#         figurePlotSave(sim_conditions, debris, sim_run_test, i)   #remember to add i for overnight runs
#         outfile = open(direc + filename + str(i) + '.pkl','wb')
#         pkl.dump({'simcond':sim_conditions,'simrun':sim_run_test,'debris':debris}, outfile)
#         outfile.close()
#         # animateTrajectory(sim_conditions, sim_run_test, debris)
#     i = i + 1