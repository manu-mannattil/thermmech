# -*- coding: utf-8 -*-

import numpy as np
from origami import *

def dyn_perp(q):
    """Determinant of the dynamical matrix restricted to the normal space."""
    j = jac(q)
    return np.sqrt(np.linalg.det(j.dot(j.T)))

# 179899 is the number of points along the longest branch.
pp = np.zeros(179899)
cv = np.linspace(0, 180, 179899)

for i in range(1, 5):
    name = "branches/fold_5-6/branch{}p.dat".format(i)
    qq = np.loadtxt(name)
    aa, gg, qq = qq[:, 0], qq[:, 1], qq[:, 2:]
    dd = np.apply_along_axis(dyn_perp, 1, qq)
    pp[:aa.size] += gg / dd

# The landscape is symmetric wrt theta, so just mirror the plot.
free = -np.log(pp/pp[-1])
cv = np.concatenate([-cv[::-1], cv[1:]])
free = np.concatenate([free[::-1], free[1:]])

np.savetxt("free_theor.dat", np.array([cv, free]).T)
