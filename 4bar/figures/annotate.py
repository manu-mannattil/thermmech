# -*- coding: utf-8 -*-
"""Plot the four-bar linkage in two configurations.

This script is for annotating the torus that represents the shape
space of the four-bar linkage.
"""

import numpy as np
import matplotlib.pyplot as plt

edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]
], dtype=int)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1)
ax.set_axis_off()

# Parallelogram branch -------------------------------------------------

alpha = 7 * np.pi / 24
parallelogram = np.array([
    [0, 0],
    [np.cos(alpha), np.sin(alpha)],
    [2 + np.cos(alpha), np.sin(alpha)],
    [2, 0]
])

for e in edges:
    x, y = parallelogram[e].T
    ax.plot(x, y, "k-")

# Twisted branch -------------------------------------------------------

beta = 17 * np.pi / 24
twisted = np.array([
    [0, 0],
    [np.cos(beta), np.sin(beta)],
    [2 + (5 * np.cos(beta) - 4)/(5 - 4 * np.cos(beta)), -3 * np.sin(beta)/(5 - 4 * np.cos(beta))],
    [2, 0]
])
twisted = twisted - np.array([0, 2])

for e in edges:
    x, y = twisted[e].T
    ax.plot(x, y, "k-")

plt.tight_layout()
plt.savefig("annotate.svg", transparent=True, bbox_inches="tight", facecolor="none", pad_inches=0)
