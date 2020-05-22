#!/usr/bin/env python

import os
import pinocchio as pin
import numpy as np
import math
np.set_printoptions(linewidth=150)

# Get the directory path for this file
dirpath = os.path.dirname(os.path.abspath(__file__))

# Import model and create data
m = pin.buildModelFromUrdf(dirpath+"/../model/urdf/box.urdf", pin.JointModelFreeFlyer())
d = m.createData()
# Set the joint configuration (x,y,z,qx,qy,qz,qw)
q = pin.utils.zero(m.nq)
q[0] = 1.
q[1] = 2.
q[2] = 0.
q[3] = 0.
q[4] = 0.
q[5] = 0.259
q[6] = 0.966
theta = math.pi/6
print(q)

# Create contact frame w.r.t. the object's CoM
object_contactId = m.addFrame(pin.Frame("object_contactPoint", m.getJointId("box_root_joint"), 0,
                              pin.SE3.Identity(), pin.FrameType.OP_FRAME))
# Set the displacement of the contact frame w.r.t. the object's CoM
lx = -0.025
ly = 0.01
lz = 0.
m.frames[object_contactId].placement.translation = np.array([lx, ly, lz])
# Perform forward kinematics
pin.computeJointJacobians(m, d, q)
pin.framesForwardKinematics(m, d, q)
# Get and print the Jacobians w.r.t. local, local-world-aligned, and world frames
Jlocal = pin.getFrameJacobian(m, d, object_contactId, pin.ReferenceFrame.LOCAL)
print(Jlocal)
Jworld = pin.getFrameJacobian(m, d, object_contactId, pin.ReferenceFrame.WORLD)
print(Jworld)
Jlocalw = pin.getFrameJacobian(m, d, object_contactId, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print(Jlocalw)

# Symbolic Jacobian
cth, sth = np.cos(theta), np.sin(theta)
Jlocalw_sym = np.array([[cth, -sth, 0., 0.,    0., -lx*sth-ly*cth],
                        [sth,  cth, 0., 0.,    0.,  lx*cth-ly*sth],
                        [0.,    0., 1., ly,   -lx,             0.],
                        [0.,    0., 0., cth, -sth,             0.],
                        [0.,    0., 0., sth,  cth,             0.],
                        [0.,    0., 0.,  0.,   0.,             1.]])
print("\nSymbolic Jacobian:")
print(Jlocalw_sym)