# -*- coding: utf-8 -*-
"""Numerical free-energy around the singularity at (0, 0, 0)."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from scipy.special import gamma
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
extent = np.pi/4
t = np.linspace(-extent, extent, N)
A = np.loadtxt("../data/free.dat")

with plt.rc_context(rc):
    fig, ax = plt.subplots()
    ax.imshow(A, cmap=bluesmoke, interpolation='nearest', origin='lower', extent=[-extent, extent, -extent, extent], rasterized=True)

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

    beta = 10 * 1000
    kappa = 1
    a = 1
    extent = np.pi / 4
    Z = np.sqrt(beta * kappa) * a / (2 * np.sqrt(5))

    # Second-order bottom of valley.
    g, pi = gamma(1/4), np.pi
    hyp = g ** 2 * (g ** 4 + 8 * pi ** 2 - np.sqrt((g ** 4 + 8 * pi ** 2) ** 2 - 256 * pi ** 4)) / (64 * Z * pi ** 3)
    ax.plot(t[:N//2], hyp / t[:N//2], "w--")
    ax.plot(t[N//2:], hyp / t[N//2:], "w--")

    plt.savefig(
        "free_num.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
