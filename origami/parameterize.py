# -*- coding: utf-8 -*-
"""Parameterize the one-dimensional branches of the origami.

This script numerically parameterizes the four one-dimensional branches
(as one-dimensional curves embedded in a twelve-dimensional ambient
space) of an origami made by triangulating a unit square.
"""

import numpy as np
import os
from origami import *
from utils import *

def hit(qq, start=50):
    """Find the point (index) when any two faces of the origami hit each other."""
    # First, find when any two faces that share an edge have an intersection.
    aa = np.apply_along_axis(angles, 1, qq)
    sign = np.array([np.sign(aa[start]) * a for a in aa])

    try:
        switch = np.where(sign < 0)[0][0]
    except IndexError as e:
        raise RuntimeError("Angle signs unchanged.  Increase 'max_steps'.")

    # Now, look at faces that share a vertex and faces that share nothing.
    faceinter = lambda A, B: triinter(A, B, False)[0]
    for i, q in enumerate(qq[start:]):
        if (
            # Faces that share a vertex.
            faceinter([[q[10], q[11], 0], [0, 0, 0], [q[0], 0, 0]],
                      [[q[10], q[11], 0], [q[1], q[2], q[3]], [q[7], q[8], q[9]]])
            or faceinter([[0, 0, 0], [q[0], 0, 0], [q[10], q[11], 0]],
                         [[0, 0, 0], [q[7], q[8], q[9]], [q[4], q[5], q[6]]])
            or faceinter([[q[1], q[2], q[3]], [q[10], q[11], 0], [q[0], 0, 0]],
                         [[q[1], q[2], q[3]], [q[4], q[5], q[6]], [q[7], q[8], q[9]]])
            or faceinter([[q[10], q[11], 0], [q[0], 0, 0], [q[1], q[2], q[3]]],
                         [[q[10], q[11], 0], [q[7], q[8], q[9]], [0, 0, 0]])
            or faceinter([[q[7], q[8], q[9]], [q[10], q[11], 0], [q[1], q[2], q[3]]],
                         [[q[7], q[8], q[9]], [q[4], q[5], q[6]], [0, 0, 0]])
            or faceinter([[q[7], q[8], q[9]], [q[1], q[2], q[3]], [q[4], q[5], q[6]]],
                         [[q[7], q[8], q[9]], [0, 0, 0], [q[10], q[11], 0]])

            # Faces that share nothing.
            or faceinter([[0, 0, 0], [q[0], 0, 0], [q[10], q[11], 0]],
                         [[q[1], q[2], q[3]], [q[4], q[5], q[6]], [q[7], q[8], q[9]]])
            or faceinter([[q[0], 0, 0], [q[1], q[2], q[3]], [q[10], q[11], 0]],
                         [[0, 0, 0], [q[7], q[8], q[9]], [q[4], q[5], q[6]]])
        ):
            break

    return min(start + i, switch)

def metric(p, qq):
    """Square-root of the metric tensor."""
    dt = np.apply_along_axis(np.linalg.norm, 1, np.diff(qq, axis=0))
    dp = np.diff(p)
    return dt / dp

# The values follow the convention (tangent, max_angle, max_steps).
# 'tangent' is the tangent to the positive branch (i.e., one where fold angle
# 5-6 is always positive).  'max_angle' is the maximum possible angle before
# two faces hit each other.  'max_steps' is a number slighter larger than the
# one required to achieve max_angle when parameterizing in arc length.
BRANCHES = {
    "branch1": (b1, 59.9, 3200),
    "branch2": (b2, 116.9, 2500),
    "branch3": (b3, 116.9, 1800),
    "branch4": (b4, 179.9, 1600),
}

bp = BranchParam(f, jac)

os.makedirs("./branches/arc/", exist_ok=True)
os.makedirs("./branches/fold_5-6", exist_ok=True)
os.chdir("./branches")

for name, info in BRANCHES.items():
    qdot0, max_angle, max_steps = info

    qq = bp.arcpar(q0=flat, qdot0=qdot0, max_steps=max_steps)
    h = hit(qq)
    np.savetxt("arc/{}p.dat".format(name), qq[:h])
    np.savetxt("arc/{}p_angles.dat".format(name), np.apply_along_axis(angles, 1, qq[:h]))
    print("positive {} in arc length done!".format(name))

    points = np.linspace(0, max_angle, int(max_angle * 1000))
    p, qq = bp.funpar(q0=flat, qdot0=qdot0, par=lambda q: angles(q, 3), points=points, arc_steps=h)
    g, p, qq = metric(p, qq), p[:-1], qq[:-1]
    np.savetxt(
        "fold_5-6/{}p.dat".format(name),
        np.hstack([p[:, np.newaxis], g[:, np.newaxis], qq]),
    )
    print("positive {} in angle of fold 5-6 done!".format(name))

    qq = bp.arcpar(q0=flat, qdot0=-qdot0, max_steps=max_steps)
    h = hit(qq)
    np.savetxt("arc/{}n.dat".format(name), qq[:h])
    np.savetxt("arc/{}n_angles.dat".format(name), np.apply_along_axis(angles, 1, qq[:h]))
    print("negative {} in arc length done!".format(name))

    points = np.linspace(-max_angle, 0, int(max_angle * 1000))
    p, qq = bp.funpar(q0=flat, qdot0=-qdot0, par=lambda q: angles(q, 3), points=points, arc_steps=h)
    g, p, qq = metric(p, qq), p[:-1], qq[:-1]
    np.savetxt(
        "fold_5-6/{}n.dat".format(name),
        np.hstack([p[:, np.newaxis], g[:, np.newaxis], qq]),
    )
    print("negative {} in angle of fold 5-6 done!".format(name))
