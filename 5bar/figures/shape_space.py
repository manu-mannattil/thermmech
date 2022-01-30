# -*- coding: utf-8 -*-
"""Visualize the shape space of the five-bar linkage."""

import numpy as np
import matplotlib.pyplot as plt
import mmmpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import axes3d
from skimage import measure

rc = {
    "mmmpl.doc": "aps",
    "mmmpl.square": 0,
    "mmmpl.tex": True,
    "mmmpl.tex.font": "fourier",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.figsize": [2.25, 2.0]
}

bluesmoke = LinearSegmentedColormap.from_list(
    "bluesmoke", [[8 / 15, 10 / 15, 12 / 15], [11 / 15, 12 / 15, 13 / 15],
                  [14 / 15, 14 / 15, 14 / 15]][::-1]
)

asp = 2.0

def pentagon(t1, t2, t3):
    return (np.cos(t1) + np.cos(t2) + np.cos(t3) - 1)**2 + (np.sin(t1) + np.sin(t2) + np.sin(t3))**2 - asp**2

t = np.linspace(-np.pi, np.pi, 128)
T1, T2, T3 = np.meshgrid(t, t, t)
F = pentagon(T1, T2, T3)

# https://stackoverflow.com/questions/4680525/plotting-implicit-equations-in-3d
verts, faces, normals, values = measure.marching_cubes(F, level=0, spacing=[np.diff(t)[0]] * 3)
verts -= np.pi # voxel to real angles

with plt.rc_context(rc):
    fig, ax = plt.subplots(
        subplot_kw={
            'projection': '3dx',
            'proj_type': 'ortho',
            'auto_add_to_figure': False
        }
    )

    ax.view_init(elev=60, azim=-45)
    ax.patch.set_alpha(0)
    ax.set_box_aspect((1.35, 1.35, 1))

    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        faces,
        verts[:, 2],
        linewidth=0,
        cmap=bluesmoke,
        rasterized=True,
        antialiased=False
    )

    ax.set_xlim3d(-np.pi - 0.5, np.pi + 0.5)
    ticks, labels = mmmpl.ticklabels(-np.pi, np.pi, 5, np.pi, divstr=r"\pi")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    ax.set_ylim3d(-np.pi - 0.5, np.pi + 0.5)
    ticks, labels = mmmpl.ticklabels(-np.pi, np.pi, 5, np.pi, divstr=r"\pi")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    ax.set_zlim3d(-np.pi - 0.5, np.pi + 0.5)
    ticks, labels = mmmpl.ticklabels(-np.pi, np.pi, 3, np.pi, divstr=r"\pi")
    ax.set_zticks(ticks)
    ax.set_zticklabels(labels)

    ax.xaxis.set_tick_params(pad=-3)
    ax.yaxis.set_tick_params(pad=-3)
    ax.zaxis.set_tick_params(pad=-5)

    # Custom labels.
    ax.text(0.25, -np.pi - 2.75, -np.pi - 1.5, r"$\zeta_1$", horizontalalignment="center")
    ax.text(np.pi + 2.75, -0.25, -np.pi - 2.75, r"$\zeta_2$", horizontalalignment="center")
    ax.text(-np.pi - 2.0, -np.pi - 2.0, -1.75, r"$\zeta_3$", horizontalalignment="center")

    plt.savefig(
        "shape_space.pdf",
        crop=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
        dpi=300,
    )
    plt.show()
