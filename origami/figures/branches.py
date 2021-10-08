# -*- coding: utf-8 -*-
"""Plot the branches of the origami."""

import numpy as np
import matplotlib.pyplot as plt
import mmmpl
from origami import *

rc = {
    "mmmpl.doc": "aps",
    "mmmpl.square": 0,
    "mmmpl.tex": True,
    "mmmpl.tex.font": "fourier",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.figsize": [2.75, 1.7]
}

# Angle ranges: -180/180, -105/105, -105/105.
i, j, k = 3, 5, 1

with plt.rc_context(rc):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

    # Axes and grids --------------------------------------------------------------

    ax.grid(True)
    ax.view_init(elev=12, azim=-60)

    # axisinfo dictionary for all axes.
    all_axinfo = {
        "tick": {
            "inward_factor": 0,
            "outward_factor": 0.25,
            "linewidth": {
                True: 0.65,
                False: 0.6
            },
        },
        "grid": {
            "color": "#cccccc",
            "linewidth": 0.6,
            "linestyle": "--"
        },
    }

    # axisinfo dictionary just for the z axis.
    z_axinfo = {
        # Use this to reposition the spine of a particular axis.  This is an
        # undocumented part of the Matplotlib API and may break any time.
        # In fact, the conventions have changed from the time of this 2018
        # StackOverflow answer: https://stackoverflow.com/a/49601745
        "juggled": (1, 2, 1),
        # Align the ticks along the y axis.
        "tickdir": 1,
    }

    ax.xaxis._axinfo.update(all_axinfo)
    ax.yaxis._axinfo.update(all_axinfo)
    ax.zaxis._axinfo.update(all_axinfo)
    ax.zaxis._axinfo.update(z_axinfo)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Borders around the panes of the plot.  Don't set an edgecolor for the
    # zaxis's pane as the xy axes spines will be overlaid on top of it.
    ax.xaxis.pane.set_edgecolor("black")
    ax.xaxis.pane.set_linewidth(plt.rcParams["axes.linewidth"])

    ax.yaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_linewidth(plt.rcParams["axes.linewidth"])

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    # Plotting -------------------------------------------------------------

    for b in range(1, 5):
        aa = np.loadtxt("../branches/arc/branch{}p_angles.dat".format(b))
        ax.plot(aa[:, i], aa[:, j], aa[:, k], "k-")

        aa = np.loadtxt("../branches/arc/branch{}n_angles.dat".format(b))
        ax.plot(aa[:, i], aa[:, j], aa[:, k], "k-")

    ax.set_xlim3d(-190, 190)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    ax.set_ylim3d(-120, 120)
    ax.set_yticks([-90, 0, 90])
    ax.set_yticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"])

    ax.set_zlim3d(-120, 120)
    ax.set_zticks([-90, -45, 0, 45, 90])
    ax.set_zticklabels([r"$-\pi/2\,$", r"$-\pi/4\,$", r"$0$", r"$\pi/4\,$", r"$\pi/2$"], ha="right")

    ax.xaxis.set_tick_params(pad=-3)
    ax.yaxis.set_tick_params(pad=-3)
    ax.zaxis.set_tick_params(pad=-6)

    # ax.set_xlabel(r"$\theta_1$", labelpad=-4)
    # ax.set_ylabel(r"$\theta_2$", labelpad=-4)
    # ax.set_zlabel(r"$\theta_3$", labelpad=-8)

    # Override matplotlib's potato-quality label placement.
    ax.text(60, -200, -175, r"$\theta_1$", horizontalalignment="center")
    ax.text(200, 30, -205, r"$\theta_2$", horizontalalignment="center")
    ax.text(-215, -200, 0, r"$\theta_3$", horizontalalignment="center")

    ax.patch.set_alpha(0)
    ax.set_box_aspect((1.35, 1.35, 1))

    plt.savefig(
        "branches.pdf",
        crop=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
