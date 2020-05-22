#!/usr/bin/env python

from __future__ import print_function
import os
import pinocchio as pin
import numpy as np
import csv
np.set_printoptions(linewidth=150, suppress=True, precision=4)


# Get the directory path for this file
dirpath = os.path.dirname(os.path.abspath(__file__))

# Parse log file
t = []
qpos = []
qvel = []
qacc = []
ctrl = []
qfrc = []
qext = []
bias = []
xfrc = []
with open(dirpath+"/../logs/traj_sawyer_push.txt") as file:
        reader = csv.reader(file, delimiter=',')
        line = 0
        for row in reader:
                if line == 0:
                        # print(', '.join(row))
                        line += 1
                elif line <= 200:
                        t.append(float(row[0]))
                        qpos.append(np.array(map(float, row[8:15])))
                        qvel.append(np.array(map(float, row[21:28])))
                        qacc.append(np.array(map(float, row[34:41])))
                        ctrl.append(np.array(map(float, row[41:48])))
                        qfrc.append(np.array(map(float, row[54:61])))
                        qext.append(np.array(map(float, row[67:74])))
                        bias.append(np.array(map(float, row[80:87])))
                        xfrc.append(np.array(map(float, row[87:93])))
                        line += 1
                else:
                        break

# Create model & data
m = pin.buildModelFromUrdf(dirpath+"/../model/urdf/sawyer.urdf")
d = m.createData()

# Initialize external forces in joint space
fext = pin.StdVec_Force()
for k in range(m.njoints):
#     print(m.joints[k])
    fext.append(pin.Force.Zero())
# fext[1].angular = ftest
print(fext[1])

for i in range(0, 15):
        print("t: ", end="")
        print(t[i])
        # Set external force
        # print("fext: ", end="")
        for j in range(1, m.njoints):
                fext[j].angular = np.array([0.,0.,qext[i+1][j-1]])
                # fext[j] = qext[i+1][j]
                # print(fext[j])
        # Evaluate forward dynamics
        pin.computeAllTerms(m, d, qpos[i], qvel[i])
        aunc = pin.aba(m, d, qpos[i], qvel[i], ctrl[i+1])
        acon = pin.aba(m, d, qpos[i], qvel[i], ctrl[i+1]+qext[i+1])
        acon_ext = pin.aba(m, d, qpos[i], qvel[i], ctrl[i+1], fext)
        # Print variables
        print("qext:     ", end="")
        print(qext[i])
        print("qfrc:     ", end="")
        print(qfrc[i])
        print("ctrl:     ", end="")
        print(ctrl[i+1])
        print("qpos:     ", end="")
        print(qpos[i])
        print("qvel:     ", end="")
        print(qvel[i])
        print("aunc:     ", end="")
        print(aunc)
        print("qacc:     ", end="")
        print(qacc[i+1])
        print("acon:     ", end="")
        print(acon)
        print("acon_ext: ", end="")
        print(acon_ext)
        # print("bias: ", end="")
        # print(bias[i+1])
        # print("pinc: ", end="")
        # print(d.nle)