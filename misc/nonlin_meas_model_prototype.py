import matplotlib.pyplot as plt
import numpy as np
import math

sc_pos = np.array([30,-74.0])
targ_pos = np.array([0,0])

mag = np.linalg.norm(sc_pos)
ang = math.atan2(sc_pos[1],sc_pos[0])
y_direc = mag*math.sin(ang)
x_direc = mag*math.cos(ang)

sc_pos_backout = np.empty(2)
sc_pos_backout[0] = mag*math.cos(ang)
sc_pos_backout[1] = mag*math.sin(ang)

print(sc_pos_backout)

# plt.figure(1)
# plt.plot(targ_pos[0],targ_pos[1],'bx')
# plt.plot(sc_pos[0],sc_pos[1],'ro')
# plt.quiver(0,0,x_direc,y_direc, angles = 'xy', scale_units = 'xy', scale = 1)
# plt.gca().set_aspect('equal')
# plt.xlim(-100,100)
# plt.ylim(-100,100)
# plt.show()