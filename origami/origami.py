# -*- coding: utf-8 -*-

"""Definitions for the origami."""

import numpy as np
from numba import njit
from utils import *

def lab(q):
    """Return the coordinates of all vertices in the lab frame."""
    return np.array([
        [0, 0, 0],
        [q[0], 0, 0],
        q[1:4],
        q[4:7],
        q[7:10],
        [q[10], q[11], 0]
    ])

# Flat-state coordinates of each vertex.
flat = np.array([1, 1, 1, 0, 0, 1, 0, 1 / 4, 1 / 2, 0, 3 / 4, 1 / 2])
real_flat = lab(flat).flatten()

# Squared-lengths of the edges.
lengths2 = np.array(
    [1.0, 1.0, 1.0, 1.0, 0.3125, 0.8125, 0.3125, 0.8125, 0.3125, 0.3125, 0.25]
)

# Natural lengths of the edges and the diagonal matrix of inverse lengths.
lengths = np.sqrt(lengths2)
lengths_diag = np.diag(1 / lengths)

# Tangents to the branches.
b1 = np.array([0., 0., 0., 0.0911365, 0., 0., 0.971228, 0., 0., 0.220023, 0., 0.])
b2 = np.array([0., 0., 0., 0.139833, 0., 0., 0.93085, 0., 0., 0.337587, 0., 0.])
b3 = np.array([0., 0., 0., -0.209162, 0., 0., 0.974036, 0., 0., 0.0866376, 0., 0.])
b4 = np.array([0., 0., 0., -0.789822, 0., 0., 0.518798, 0., 0., 0.327155, 0., 0.])

# Faces of the origami (6 triangles).
faces = np.array([
    [0, 1, 5],
    [1, 2, 5],
    [2, 4, 5],
    [5, 4, 0],
    [2, 3, 4],
    [3, 0, 4]
])

@njit(fastmath=True)
def f(q):
    """Constraint map for the origami."""
    return 0.5 * (np.array([
        q[0] ** 2,
        (q[1] - q[0]) ** 2 + q[2] ** 2 + q[3] ** 2,
        (q[4] - q[1]) ** 2 + (q[5] - q[2]) ** 2 + (q[6] - q[3]) ** 2,
        q[4] ** 2 + q[5] ** 2 + q[6] ** 2,
        q[7] ** 2 + q[8] ** 2 + q[9] ** 2,
        (q[7] - q[1]) ** 2 + (q[8] - q[2]) ** 2 + (q[9] - q[3]) ** 2,
        (q[7] - q[4]) ** 2 + (q[8] - q[5]) ** 2 + (q[9] - q[6]) ** 2,
        q[10] ** 2 + q[11] ** 2,
        (q[10] - q[0]) ** 2 + q[11] ** 2,
        (q[10] - q[1]) ** 2 + (q[11] - q[2]) ** 2 + q[3] ** 2,
        (q[10] - q[7]) ** 2 + (q[11] - q[8]) ** 2 + q[9] ** 2,
    ]) - lengths2) / (lengths)

@njit(fastmath=True)
def jac(q):
    """Jacobian of the origami."""
    # Why is this a mess?  Because Numba can't speed up complicated definitions.
    return lengths_diag.dot(np.array([
        [q[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [q[0] - q[1], -q[0] + q[1], q[2], q[3], 0, 0, 0, 0, 0, 0, 0, 0],
        [0, q[1] - q[4], q[2] - q[5], q[3] - q[6], -q[1] + q[4], -q[2] + q[5], -q[3] + q[6], 0, 0, 0, 0, 0],
        [0, 0, 0, 0, q[4], q[5], q[6], 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, q[7], q[8], q[9], 0, 0],
        [0, q[1] - q[7], q[2] - q[8], q[3] - q[9], 0, 0, 0, -q[1] + q[7], -q[2] + q[8], -q[3] + q[9], 0, 0],
        [0, 0, 0, 0, q[4] - q[7], q[5] - q[8], q[6] - q[9], -q[4] + q[7], -q[5] + q[8], -q[6] + q[9], 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, q[10], q[11]],
        [q[0] - q[10], 0, 0, 0, 0, 0, 0, 0, 0, 0, -q[0] + q[10], q[11]],
        [0, q[1] - q[10], q[2] - q[11], q[3], 0, 0, 0, 0, 0, 0, -q[1] + q[10], -q[2] + q[11]],
        [0, 0, 0, 0, 0, 0, 0, q[7] - q[10], q[8] - q[11], q[9], -q[7] + q[10], -q[8] + q[11]]
    ]))

def angles(q, a=None, dihedral=False):
    """Return the angle of each fold."""
    r = lab(q)

    if a == 0:
        return normangle(r[2] - r[4], r[5] - r[4], r[3] - r[4], dihedral) # fold 3-5
    elif a == 1:
        return normangle(r[3] - r[4], r[2] - r[4], r[0] - r[4], dihedral) # fold 4-5
    elif a == 2:
        return normangle(r[0] - r[4], r[3] - r[4], r[5] - r[4], dihedral) # fold 1-5
    elif a == 3:
        return normangle(r[5] - r[4], r[0] - r[4], r[2] - r[4], dihedral) # fold 6-5
    elif a == 4:
        return normangle(r[0] - r[5], r[4] - r[5], r[1] - r[5], dihedral) # fold 1-6
    elif a == 5:
        return normangle(r[1] - r[5], r[0] - r[5], r[2] - r[5], dihedral) # fold 2-6
    elif a == 6:
        return normangle(r[2] - r[5], r[1] - r[5], r[4] - r[5], dihedral) # fold 3-6
    else:
        return np.array([
            normangle(r[2] - r[4], r[5] - r[4], r[3] - r[4], dihedral),   # fold 3-5
            normangle(r[3] - r[4], r[2] - r[4], r[0] - r[4], dihedral),   # fold 4-5
            normangle(r[0] - r[4], r[3] - r[4], r[5] - r[4], dihedral),   # fold 1-5
            normangle(r[5] - r[4], r[0] - r[4], r[2] - r[4], dihedral),   # fold 6-5
            normangle(r[0] - r[5], r[4] - r[5], r[1] - r[5], dihedral),   # fold 1-6
            normangle(r[1] - r[5], r[0] - r[5], r[2] - r[5], dihedral),   # fold 2-6
            normangle(r[2] - r[5], r[1] - r[5], r[4] - r[5], dihedral),   # fold 3-6
        ])
