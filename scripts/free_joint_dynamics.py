#!/usr/bin/env python3

import os
import math
import numpy as np
import pinocchio as pin
np.set_printoptions(linewidth=150, precision=6, suppress=True)

def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])

# Get the directory path for this file
dirpath = os.path.dirname(os.path.abspath(__file__))

# Import model and create data
# m = pin.buildModelFromUrdf(dirpath+"/../model/urdf/red_box.urdf", pin.JointModelFreeFlyer())
m = pin.buildModelFromUrdf(dirpath+"/../model/urdf/red_box.urdf")
d = m.createData()
# Set the configuration [x,y,z,qx,qy,qz,qw]
q = pin.utils.zero(m.nq)
q[0] = 1.10264
q[1] = -0.000452574
q[2] = 0.0660566
q[3] = -0.00127484
q[4] = -0.000904901
q[5] = 0.00206843
q[6] = 0.999997
# Set the velocities
v = pin.utils.zero(m.nv)
v[0] = 0.528398
v[1] = -0.0905147
v[2] = 0.325857
v[3] = -0.509935
v[4] = -0.361961
v[5] = 0.827372
# Set the joint forces
u = pin.utils.zero(m.nv)

# Random state and control
# q = pin.utils.rand(m.nq)
# q[3:7] = q[3:7]/np.linalg.norm(q[3:7])
# v = pin.utils.rand(m.nv)
# u = pin.utils.rand(m.nv)

# Set the external forces
fext = pin.StdVec_Force()
for k in range(m.njoints):
    fext.append(pin.Force.Zero())
# fext[1].linear = np.array([13.533, -1.20753, -3.09618])
# fext[1].angular = np.array([0.0259438, 0.0685129, 0.0866765])


# Calculate the accelerations and other terms
pin.computeAllTerms(m, d, q, v)
a = pin.aba(m, d, q, v, u, fext)

# Print state and controls
print("pos =", q)
print("vel =", v)
print("tau =", u)
print("acc =", a)
print("nle =", d.nle)

# Get the joint ID for the object
# obj_jnt_id = m.getJointId("object")
obj_jnt_id = 1
# print("Object joint id: ", obj_jnt_id)
# Calculate the classical accelerations for the object
pin.forwardKinematics(m, d, q, v, a)
acc_spatial = pin.getAcceleration(m, d, obj_jnt_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
acc_classic = pin.getClassicalAcceleration(m, d, obj_jnt_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
vel_spatial = pin.getVelocity(m, d, obj_jnt_id, pin.LOCAL_WORLD_ALIGNED)
acc_diff = np.cross(vel_spatial.linear, vel_spatial.angular)
# print("\nUsing Pinocchio getters:")
# print("Spatial acceleration:\n", acc_spatial)
# print("Classic acceleration:\n", acc_classic)
# print("Difference between spatial and classical accelerations:")
# print("v x w =", acc_diff)

a_world = d.oMi[obj_jnt_id].act(d.a[obj_jnt_id])
R = d.oMi[obj_jnt_id].rotation
acc_spatial_v_lwa = R.dot(d.a[obj_jnt_id].linear)
acc_classic_v_lwa = acc_spatial_v_lwa - acc_diff
acc_spatial_w = d.a[obj_jnt_id].angular
acc_classic_w = d.a[obj_jnt_id].angular
# print("\nUsing transformations:")
# print("Spatial acceleration in W:\n", a_world)
# print("Spatial acceleration in LWA:", acc_spatial_v_lwa, acc_spatial_w)
# print("Classic acceleration in LWA:", acc_classic_v_lwa, acc_classic_w)

# Evaluate the derivatives of ABA
pin.computeABADerivatives(m, d, q, v, u, fext)
print("\nABA derivatives:")
print("ddq_dq:\n", d.ddq_dq)
print("ddq_dv:\n", d.ddq_dv)

# dIntegrate
# Jq, Jv = pin.dIntegrate(m, q, v)
# print("\nJq:\n", Jq)
# print("Jv:\n", Jv)

# # Joint velocity derivatives
# pin.computeForwardKinematicsDerivatives(m, d, q, v, a)
# dv = pin.getJointVelocityDerivatives(m, d, obj_jnt_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
# print("\ndv_dq:\n", dv[0])
# print("dv_dv:\n", dv[1])

# # Joint acceleration derivatives
# da = pin.getJointAccelerationDerivatives(m, d, obj_jnt_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
# print("\ndv_dq:\n", da[0])
# print("da_dq:\n", da[1])
# print("da_dv:\n", da[2])
# print("da_da:\n", da[3])

# Velocities
# v_world = d.oMi[obj_jnt_id].act(d.v[obj_jnt_id])
# v_w = pin.getVelocity(m, d, obj_jnt_id, pin.ReferenceFrame.WORLD)
# v_l = pin.getVelocity(m, d, obj_jnt_id, pin.ReferenceFrame.LOCAL)
# v_lwa = pin.getVelocity(m, d, obj_jnt_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
# print("\nv_w:\n", v_w)
# print("\nv_l:\n", v_l)
# print("\nv_lwa:\n", v_lwa)

# Integration
# q_int = pin.integrate(m, q, v)
# q_diff = q_int - q
# print("\nIntegration:")
# print("q =", q)
# print("v =", v)
# print("q_int =", q_int)
# print("translation =", q_diff[0:3])

# Exponential of the spatial motion
# t2 = np.inner(v_l.angular, v_l.angular)
# t = np.sqrt(t2)
# ct, st = math.cos(t), math.sin(t)
# t2inv = 1/t2
# if t < 1e-3:
#     alpha_wxv = 1/2 - t2/24
#     alpha_v = 1 - t2/6
#     alpha_w = 1/6 - t2/120
#     diagonal = 1 - t2/2
# else:
#     alpha_wxv = (1-ct)*t2inv
#     alpha_v = st/t
#     alpha_w = (1-alpha_v)*t2inv
#     diagonal = ct
# trans = alpha_v*v_lwa.linear + alpha_w*v_lwa.angular.dot(v_lwa.linear)*v_lwa.angular + alpha_wxv*np.cross(v_lwa.angular, v_lwa.linear)
# print(pin.exp6(v_lwa))
# print("\nt, t2, t2inv:", t, t2, t2inv)
# print("ct, st:", ct, st)
# print("alpha wxv, v, w, diag:", alpha_wxv, alpha_v, alpha_w, diagonal)
# print("trans:", trans)

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