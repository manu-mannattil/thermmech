#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parameterize a unit circle in R^2 numerically."""

import matplotlib.pyplot as plt
import numpy as np
from utils import *

def f_circle(q):
    """Constraint equation that defines a circle."""
    return np.array([np.sum(q ** 2) - 1])


def jac_circle(q):
    """Jacobian of f_circle()."""
    return 2 * np.array([q])


def par(q):
    """Parameterize using t^3 + t, where t is the polar angle."""
    t = np.arctan2(q[1], q[0])

    # A simple diffeomorphism from R^1 -> R^1.
    return t ** 3 + t


q0 = np.array([1, 0])
qdot0 = np.array([0, 1])

bp = BranchParam(f_circle, jac_circle)

qq = bp.arcpar(q0, qdot0, max_steps=629, step_length=1e-2)
_, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_title(r"Circle parameterized by $\theta$")
ax.plot(qq[::25, 0], qq[::25, 1], "o", markersize=3)

pp, qq = bp.funpar(q0, qdot0, par, arc_steps=629, step_length=1e-2)
_, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_title(r"Circle parameterized by $\theta^3 + \theta$")
ax.plot(qq[::25, 0], qq[::25, 1], "o", markersize=3)

plt.show()
