# -*- coding: utf-8 -*-
"""Verify that the CV map and its linearization is correct.

We need to explicitly verify that the expression for the CV map (and its
linearization) gives the correct fold angles.  The fold angles in the
parameterization data files were computed using the normangle() function, which
computes it as the angle between the normals of the faces (which turns out to
be a mathematically complicated expression to manipulate).  Here we use
simple high-school geometry to compute fold angle 5-6, which is our CV.

Also show the 'extent' of the positive branches in fold angles:

branch  fold_3-5       fold_4-5      fold_1-5       fold_6-5      fold_1-6       fold_2-6      fold_3-6
------  --------       --------      --------       --------      --------       --------      --------
1       -179.93247609  -84.73458147  -122.57893407  116.94716405  -179.93247609  -84.73458147  -122.57893407
2       -72.64940922   84.74565655   62.60208159    116.96696051  -179.94757616  -84.74565655  -122.5789485
3       -179.93069015  -84.73327145  -122.57893212  116.94482268  -72.63161898   84.73327145   62.58736895
4       -128.5999345   104.47746529  104.42647763   179.90875901  -128.5999345   104.47746529  104.42647763
"""

import numpy as np
import matplotlib.pyplot as plt
from origami import *

def cv(q):
    """CV computed exactly for angles <= 90."""
    # Coordinates of joints 3, 5, and 6.
    j3 = np.array([q[1], q[2], q[3]])
    j5 = np.array([q[7], q[8], q[9]])
    j6 = np.array([q[10], q[11], 0])

    # Normal to the plane 1-5-6.
    m = np.cross(j5, j6)
    n = m / np.linalg.norm(m)

    # Perpendicular distance from joint 3 to the plane 1-5-6.
    h = n.dot(j3)

    # Tangent along fold 5-6.
    t = j6 - j5
    t /= np.linalg.norm(t)

    # Parameter giving the perpendicular distance from the projection of
    # joint 3 on the plane 1-5-6 to fold 5-6.
    s = -q[3] * q[8] * q[10] + q[2] * q[9] * q[10] + q[3] * q[7] * q[11] - q[1] * q[9] * q[11]
    s /= (
        q[8]**2 * q[10]**2 + q[9]**2 * q[10]**2 - 2 * q[7] * q[8] * q[10] * q[11] +
        q[7]**2 * q[11]**2 + q[9]**2 * q[11]**2
    )

    # Perpendicular distance from projection to fold.
    # This fails for > 90 angles since we're not using the signed distance.
    d = np.linalg.norm(np.cross(j3 - j5 - s*m, t))

    return np.arctan(h / d) * 180 / np.pi

def lincv(q):
    """Linearized CV."""
    return 2 * (q[9] - q[3]) * 180 / np.pi

for i in range(1, 5):
    name = "branches/arc/branch{}p.dat".format(i)
    qq = np.loadtxt(name)
    print(r"Branch #{} extent: ".format(i), angles(qq[-1]))
    _, ax = plt.subplots()
    ax.set_title(r"Branch #{}".format(i))
    ax.plot(np.apply_along_axis(lambda q: angles(q, 3), 1, qq[:500]), label="measured")
    ax.plot(np.apply_along_axis(cv, 1, qq[:500]), label="computed")
    ax.plot(np.apply_along_axis(lincv, 1, qq[:500]), label="linearized")
    ax.legend()

plt.show()
