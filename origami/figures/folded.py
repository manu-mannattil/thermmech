# -*- coding: utf-8 -*-
"""Make folded origami figures for the end of each branch."""

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from origami import *

hue, val, sat_min, sat_max = 0.58, 0.2, 0.7, 1.0

fig = plt.figure()

ax = fig.add_subplot(projection="3d", proj_type="ortho")
qq = np.loadtxt("../branches/arc/branch1n.dat")[-1]
plot_origami(lab(qq), faces, axis=ax, hue=hue, val=val, sat_min=sat_min, sat_max=sat_max)
plt.savefig("folded_1.svg", transparent=True)
ax.clear()

qq = np.loadtxt("../branches/arc/branch2n.dat")[-1]
plot_origami(lab(qq), faces, axis=ax, hue=hue, val=val, sat_min=sat_min, sat_max=sat_max)
plt.savefig("folded_2.svg", transparent=True)
ax.clear()

qq = np.loadtxt("../branches/arc/branch3p.dat")[-1]
plot_origami(lab(qq), faces, axis=ax, hue=hue, val=val, sat_min=sat_min, sat_max=sat_max)
plt.savefig("folded_3.svg", transparent=True)
ax.clear()

qq = np.loadtxt("../branches/arc/branch4p.dat")[-1]
plot_origami(lab(qq), faces, axis=ax, hue=hue, val=val, sat_min=sat_min, sat_max=sat_max)
plt.savefig("folded_4.svg", transparent=True)
ax.clear()
