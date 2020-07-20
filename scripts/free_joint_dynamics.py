#!/usr/bin/env python3

import os
import math
import numpy as np
import pinocchio as pin
np.set_printoptions(linewidth=150)

# Get the directory path for this file
dirpath = os.path.dirname(os.path.abspath(__file__))

# Import model and create data
# m = pin.buildModelFromUrdf(dirpath+"/../model/urdf/red_box.urdf", pin.JointModelFreeFlyer())
m = pin.buildModelFromUrdf(dirpath+"/../model/urdf/red_box.urdf")
d = m.createData()
# Set the configuration [x,y,z,qx,qy,qz,qw]
q = pin.utils.zero(m.nq)
q[0] = 1.
q[1] = 2.
q[2] = 0.
q[3] = 0.
q[4] = 0.259
q[5] = 0.
q[6] = 0.966
print("pos =", q)
# Set the velocities
v = pin.utils.zero(m.nv)
# v[0] = 0.528398
# v[1] = -0.0905147
# v[2] = 0.325857
# v[3] = -0.509935
# v[4] = -0.361961
# v[5] = 0.827372
print("vel =", v)
# Set the forces
u = pin.utils.zero(m.nv)
# Calculate the accelerations
pin.nonLinearEffects(m, d, q, v)
a = pin.aba(m, d, q, v, u)
print("acc =", a)
print("nle =", d.nle)

# Get the joint ID for the object
obj_jnt_id = m.getJointId("object")
print("Object joint id: ", obj_jnt_id)
# Calculate the classical accelerations for the object
pin.forwardKinematics(m, d, q, v, a)
acc_spatial = pin.getAcceleration(m, d, obj_jnt_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
acc_classic = pin.getClassicalAcceleration(m, d, obj_jnt_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
vel_spatial = pin.getVelocity(m, d, obj_jnt_id, pin.LOCAL_WORLD_ALIGNED)
# Print results
print("Acceleration in data:\n", d.a[1])
print("Spatial acceleration:\n", acc_spatial)
print("Classical acceleration:\n", acc_classic)
print("Difference between spatial and classical accelerations:")
print("v x w =", np.cross(vel_spatial.linear, vel_spatial.angular))

# Evaluate the derivatives of ABA
pin.computeABADerivatives(m, d, q, v, u)
print("\nABA derivatives:")
print("ddq_dq:\n", d.ddq_dq)
print("ddq_dv:\n", d.ddq_dv)

# dIntegrate
J = pin.dIntegrate(m, q, v, pin.ArgumentPosition.ARG0)
print("\nJ:\n", J)

# Joint velocity derivatives
dv = pin.getJointVelocityDerivatives(m, d, obj_jnt_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print("\ndv_dq:\n", dv[0])
print("\ndv_dv:\n", dv[1])

# Compute various dynamic quantities 
# pin.computeCoriolisMatrix(m, d, q, v)
# c = d.C.dot(v)
# print("C:\n", d.C)
# print("c =", c)
# print("g =", d.nle-c)
# pin.ccrba(m, d, q, v)
# print("Ag:\n", d.Ag)
# print("J:\n", d.J)
# Potential energy
# pin.computePotentialEnergy(m, d, q)
# print("P = ", d.potential_energy)
# Kinetic energy
# pin.computeKineticEnergy(m, d, q, v)
# print("K = ", d.kinetic_energy)
# Center of mass
# pin.centerOfMass(m, d, q, v)
# print("com = ", d.com[0])
# print("com = ", d.vcom[0])