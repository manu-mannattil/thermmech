# -*- coding: utf-8 -*-
"""Theoretical free-energy around the singularity at (0, 0, 0)."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from scipy.special import pbdv, gamma
from matplotlib.colors import LinearSegmentedColormap

bluesmoke = LinearSegmentedColormap.from_list(
    "bluesmoke", [[4 / 15, 7 / 15, 9 / 15], [8 / 15, 10 / 15, 12 / 15], [11 / 15, 12 / 15, 13 / 15],
                  [14 / 15, 14 / 15, 14 / 15], [15 / 15, 15 / 15, 15 / 15]]
)

rc = {
    "charu.doc": "aps",
    "charu.square": 0,
    "charu.tex.font": "fourier",
    "charu.tex": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.figsize": [1.8, 1.8]
}

N = 128

beta = 10 * 1000
kappa = 1
a = 1

extent = np.pi / 4
Z = np.sqrt(beta * kappa) * a / (2 * np.sqrt(5))

t = np.linspace(-extent, extent, N)
z1, z2 = np.meshgrid(t, t)
A = Z**2 * z1**2 * z2**2 - np.log(pbdv(-1 / 2, -2 * Z * z1 * z2)[0]) + np.log(pbdv(-1 / 2, 0)[0])

# 5.48 is the maximum free energy from numerics.
A = np.ma.masked_where(A > 5.48, A)

with plt.rc_context(rc):
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(z1, z2, A, cmap=bluesmoke, norm=None, shading="nearest", rasterized=True)
    # fig.colorbar(pcm, ax=ax, location='bottom')

    ax.set_xlabel(r"$\zeta_1$", rotation=0)
    ticks, labels = charu.ticklabels(-extent, extent, 5, np.pi, divstr=r"\pi")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim((-extent, extent))

    ax.set_ylabel(r"$\zeta_2$", labelpad=0, rotation=0)
    ticks, labels = charu.ticklabels(-extent, extent, 5, np.pi, divstr=r"\pi")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim((-extent, extent))

    # First-order bottom of valley.
    # hyp = 2 * np.pi * gamma(1/4) ** 2 / (Z * (8 * np.pi ** 2 + gamma(1/4) * 4))

    # Second-order bottom of valley.
    g, pi = gamma(1/4), np.pi
    hyp = g ** 2 * (g ** 4 + 8 * pi ** 2 - np.sqrt((g ** 4 + 8 * pi ** 2) ** 2 - 256 * pi ** 4)) / (64 * Z * pi ** 3)
    ax.plot(t[:N//2], hyp / t[:N//2], "w--")
    ax.plot(t[N//2:], hyp / t[N//2:], "w--")

    plt.savefig(
        "free_theor.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
