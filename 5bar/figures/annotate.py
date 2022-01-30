# -*- coding: utf-8 -*-
"""Plot configurations of the five-bar linkage."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

asp = 2

def pentagon(x, t1, t2):
    return (np.cos(t1) + np.cos(t2) + np.cos(x) - 1)**2 + (np.sin(t1) + np.sin(t2) + np.sin(x))**2 - asp**2

def pentagon_deriv(x, t1, t2):
    return 2 * ((np.sin(t1) + np.sin(t2) + np.sin(x)) * np.cos(x) -
                (np.cos(t1) + np.cos(t2) + np.cos(x)) * np.sin(x))

def solve(t1, t2, guess=[0]):
    return root(pentagon, x0=guess, args=(t1, t2), jac=pentagon_deriv).x[0]

t1, t2 = 0.628, 0.392
t3 = solve(t1, t2, [1.82])

x = [0, np.cos(t1), np.cos(t2), np.cos(t3)]
y = [0, np.sin(t1), np.sin(t2), np.sin(t3)]
x, y = np.concatenate([np.cumsum(x), [1, 0]]), np.concatenate([np.cumsum(y), [0, 0]])

print("t3 = {}".format(t3))
print("error = {}%".format(np.abs(np.sqrt((x[-3] - 1)**2 + y[-3]**2) / asp - 1) * 100))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1)
ax.set_axis_off()
ax.plot(x, y, "k-")

plt.tight_layout()
plt.savefig("annotate.svg", transparent=True, bbox_inches="tight", facecolor="none", pad_inches=0)
plt.show()
