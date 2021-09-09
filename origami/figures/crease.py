# -*- coding: utf-8 -*-
"""Plot the "crease" pattern of the origami."""

import numpy as np
import matplotlib.pyplot as plt

# Boundary vertices are on a unit square on the xy plane.
# Internal vertices are at (a, b, 0) and (c, d, 0).
a, b, c, d = 1 / 4, 0.5, 3 / 4, 0.5
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [a, b, 0],
    [c, d, 0],
], dtype=float)

edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 0],
    [4, 2],
    [4, 3],
    [5, 0],
    [5, 1],
    [5, 2],
    [5, 4],
], dtype=int)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1)
ax.set_axis_off()

for e in edges:
    x, y, _ = vertices[e].T
    ax.plot(x, y, "k-")

plt.tight_layout()
plt.savefig("crease.svg", transparent=True)
