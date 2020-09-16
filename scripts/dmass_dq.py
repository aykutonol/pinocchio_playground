#!/usr/bin/env python3

import os
import sys
import math
import numpy as np
import pinocchio as pin
np.set_printoptions(linewidth=150, precision=9, suppress=True)

def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])

# Get the directory path for this file
dirpath = os.path.dirname(os.path.abspath(__file__))

if __name__=="__main__":
    use_free_jnt = False
    try:
        if int(sys.argv[1])>0:
            use_free_jnt = True
    except:
        pass
    # Import model and create data
    if use_free_jnt:
        m = pin.buildModelFromUrdf(dirpath+"/../model/urdf/red_box.urdf")
    else:
        m = pin.buildModelFromUrdf(dirpath+"/../model/urdf/ur3e.urdf")
    d = m.createData()
    q = pin.utils.zero(m.nq)
    v = pin.utils.zero(m.nv)
    u = pin.utils.zero(m.nv)

    # Random states and control
    q = pin.utils.rand(m.nq)
    if use_free_jnt:
        q[3:7] = q[3:7]/np.linalg.norm(q[3:7])  # normalize quaternion if free joint
    v = pin.utils.rand(m.nv)
    u = pin.utils.rand(m.nv)

    # Set the external forces
    fext = pin.StdVec_Force()
    for k in range(m.njoints):
        fext.append(pin.Force.Zero())

    # Evaluate all terms and forward dynamics
    pin.computeAllTerms(m, d, q, v)
    a = pin.aba(m, d, q, v, u, fext)

    # Print state and controls
    print("pos =", q)
    print("vel =", v)
    print("tau =", u)
    print("acc =", a)
    print("nle =", d.nle)

    # Evaluate the ABA derivatives
    pin.computeABADerivatives(m, d, q, v, u, fext)
    print("\nABA derivatives:")
    print("ddq_dq:\n", d.ddq_dq)
    print("ddq_dv:\n", d.ddq_dv)
    print("M:\n", d.M)
    print("Minv:\n", d.Minv)

    # Evaluate the sensitivities of M & Minv to configuration perturbations
    M = d.M
    Minv = d.Minv
    dM_dq = np.zeros((m.nv, m.nv))
    eps = 1e-6
    for i in range(m.nv):
        q[i] += eps
        print("pos =", q)
        pin.computeAllTerms(m, d, q, v)
        pin.computeMinverse(m, d, q)
        print("newM - M\n", d.M-M)
        print("newMinv - Minv:\n", d.Minv-Minv)
        q[i] -= eps