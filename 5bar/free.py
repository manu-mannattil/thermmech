# -*- coding: utf-8 -*-

"""Find the free energy from MC data."""

import numpy as np

extent = np.pi/4

N = 72
bins = np.linspace(-extent, extent, N)

aa = np.loadtxt("data/5bar_met.dat")
z1, z2 = aa[:, 0], aa[:, 1]

A, xedges, yedges = np.histogram2d(z1, z2, bins=bins, density=False)
mid = int((N - 1) / 2)
A = -np.log(A) + np.log(A[mid][mid])
np.savetxt("data/free.dat", A)
