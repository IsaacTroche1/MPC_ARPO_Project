from src.mpcsim import *
from src.trajectorySimulate import trajectorySimulate
from src.animateTrajectory import animateTrajectory
from scipy import sparse
from matplotlib import pyplot as plt

#debris setup
center = (40.,0.)
side_length = 5.
detect_dist = 20

#noise setup
sig_x = 0.3
sig_y = 0.3
noise_length = 50

#success conditions setup
distance_tolerance = 0.2
ang_tolerance = 45

#general sim setup
platform_radius = 2.5
tolerance_radius = 1.5
los_angle = 10*(np.pi/180)
mean_motion = 1.107e-3
sample_time = 0.5
in_track = False
x0 = np.array([100.,10.,0.,0])
rx = platform_radius
ry = 0
xr = np.array([rx,ry,0.,0.])

is_deltav = False
success_cond = (distance_tolerance, ang_tolerance)
noises = Noise((sig_x,sig_y), noise_length)
# noises = None

#MPC controller setup
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

#failsafe controller setup
Q_failsafe = 0.005*np.diag([0.0001, 1, 100000., 1., 0.01])
R_failsafe = 100*np.diag([1, 1])
C_refx = np.eye(1,4)

#populate conditions
sim_conditions_norej = SimConditions(x0, xr, platform_radius, los_angle, tolerance_radius, mean_motion, sample_time, False, success_cond, noises, in_track, T_final=300, isDeltaV=is_deltav)
sim_conditions_rej = SimConditions(x0, xr, platform_radius, los_angle, tolerance_radius, mean_motion, sample_time, True, success_cond, noises, in_track, T_final=300, isDeltaV=is_deltav)
mpc_params = MPCParams(Q_mpc, R_mpc, R_mpc_s, v_ecr, horizons, ulim)
debris = Debris(center, side_length, detect_dist)
# debris = None
fail_params = FailsafeParams(Q_failsafe,R_failsafe,C_refx,np.zeros([2,2]))

MCnum = 300
success_count = 0
for j in range(MCnum):
    print(j)
    sim_run_rej = trajectorySimulate(sim_conditions_rej, mpc_params, fail_params, debris)
    if (sim_run_rej.isSuccess):
        success_count += 1

    # errNon[j] = 2
    # errComp[j] = 1

print(success_count)


