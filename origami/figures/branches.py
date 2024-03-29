# -*- coding: utf-8 -*-
"""Plot the branches of the origami."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from origami import *

rc = {
    "charu.doc": "aps",
    "charu.square": 0,
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.figsize": [2.75, 1.7]
}

# Angle ranges: -180/180, -105/105, -105/105.
i, j, k = 3, 5, 1

with plt.rc_context(rc):
    fig, ax = plt.subplots(
        subplot_kw={
            'projection': '3dx',
            'proj_type': 'ortho',
            'auto_add_to_figure': False
        }
    )

    ax.view_init(elev=12, azim=-60)

    for b in range(1, 5):
        aa = np.loadtxt("../branches/arc/branch{}p_angles.dat".format(b))
        ax.plot(aa[:, i], aa[:, j], aa[:, k], "k-")

        aa = np.loadtxt("../branches/arc/branch{}n_angles.dat".format(b))
        ax.plot(aa[:, i], aa[:, j], aa[:, k], "k-")

    ax.set_xlim3d(-190, 190)
    ticks, labels = charu.ticklabels(-np.pi, np.pi, 5, np.pi, divstr=r"\pi")
    ax.set_xticks(ticks * 180 / np.pi)
    ax.set_xticklabels(labels)

    ax.set_ylim3d(-120, 120)
    ticks, labels = charu.ticklabels(-np.pi/2, np.pi/2, 3, np.pi, divstr=r"\pi")
    ax.set_yticks(ticks * 180 / np.pi)
    ax.set_yticklabels(labels)

    ax.set_zlim3d(-120, 120)
    ticks, labels = charu.ticklabels(-np.pi/2, np.pi/2, 5, np.pi, divstr=r"\pi")
    ax.set_zticks(ticks * 180 / np.pi)
    ax.set_zticklabels(labels)

    ax.xaxis.set_tick_params(pad=-3)
    ax.yaxis.set_tick_params(pad=-3)
    ax.zaxis.set_tick_params(pad=-6)

    # ax.set_xlabel(r"$\rho_1$", labelpad=-4)
    # ax.set_ylabel(r"$\rho_2$", labelpad=-4)
    # ax.set_zlabel(r"$\rho_3$", labelpad=-8)

    # Override matplotlib's potato-quality label placement.
    ax.text(60, -200, -175, r"$\rho_1$", horizontalalignment="center")
    ax.text(200, 30, -205, r"$\rho_2$", horizontalalignment="center")
    ax.text(-215, -200, 0, r"$\rho_3$", horizontalalignment="center")

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
