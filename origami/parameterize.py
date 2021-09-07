# -*- coding: utf-8 -*-

import numpy as np
import os
from origami import *
from utils import *

def hit(qq, check_against=5):
    """Find the point when any two faces of the origami hit each other."""
    aa = np.apply_along_axis(angles, 1, qq)
    sign = np.array([np.sign(aa[check_against]) * a for a in aa])
    try:
        switch = np.where(sign < 0)[0][0]
    except IndexError as e:
        raise RuntimeError("Angle signs unchanged.  Increase 'max_steps'.")

    return switch

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
    "branch1": (b1, 116.9, 3200),
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
