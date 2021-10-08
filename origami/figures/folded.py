# -*- coding: utf-8 -*-
"""Make folded origami figures for the end of each branch."""

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from origami import *

hue, val, sat_min, sat_max, alpha = 0.58, 0.2, 0.7, 1.0, 0.8

fig = plt.figure()
ax = fig.add_subplot(projection="3d", proj_type="ortho")

ax.view_init(azim=-45, elev=120)
qq = np.loadtxt("../branches/arc/branch1n.dat")[-1]
plot_origami(lab(qq), faces, axis=ax, hue=hue, val=val, sat_min=sat_min, sat_max=sat_max, alpha=alpha)
plt.savefig("folded_1.svg", transparent=True)
ax.clear()

ax.view_init(azim=160, elev=140)
qq = np.loadtxt("../branches/arc/branch2p.dat")[-1]
plot_origami(lab(qq), faces, axis=ax, hue=hue, val=val, sat_min=sat_min, sat_max=sat_max, alpha=alpha)
plt.savefig("folded_2.svg", transparent=True)
ax.clear()

ax.view_init(azim=-90, elev=30)
qq = np.loadtxt("../branches/arc/branch3p.dat")[-1]
plot_origami(lab(qq), faces, axis=ax, hue=hue, val=val, sat_min=sat_min, sat_max=sat_max, alpha=alpha)
plt.savefig("folded_3.svg", transparent=True)
ax.clear()

ax.view_init(azim=-40, elev=15)
qq = np.loadtxt("../branches/arc/branch4p.dat")[-1]
plot_origami(lab(qq), faces, axis=ax, hue=hue, val=val, sat_min=sat_min, sat_max=sat_max, alpha=alpha)
plt.savefig("folded_4.svg", transparent=True)
ax.clear()
