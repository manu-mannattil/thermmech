# -*- coding: utf-8 -*-
"""Energy level sets of bar 3-4."""

import numpy as np
import matplotlib.pyplot as plt
import mmmpl

def energy(x, y):
    """Energy of bar 3-4."""
    return ((2 + np.cos(y) - np.cos(x)) ** 2 + (np.sin(y) - np.sin(x)) ** 2 - 4) ** 2 / 32

rc = {
    "mmmpl.doc": "aps",
    "mmmpl.square": True,
    "mmmpl.tex": True,
    "mmmpl.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    x = np.linspace(-np.pi/4, np.pi/4, 500)
    y = np.linspace(-np.pi/4, np.pi/4, 500)
    X, Y = np.meshgrid(x, y)

    levels = [0, 8e-4, 1e-2, 5e-2, 5e-1]
    colors= ["#cddeeeff", "#bdd5eaff", "#aecbe5ff", "#9ec1e0ff"]
    ax.contourf(X, Y, energy(X, Y), levels, colors=colors)
    ax.contour(X, Y, energy(X, Y), levels, linestyles="dashed", colors="#2e5c8fff")

    ax.set_xlabel(r"$\theta_1$")
    ax.set_xticks([-np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4])
    ax.set_xticklabels([r"$-\pi/4$", r"$-\pi/8$", r"$0$", r"$\pi/8$", r"$\pi/4$"])
    ax.set_ylabel(r"$\theta_2$", labelpad=0, rotation=0)
    ax.set_yticks([-np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4])
    ax.set_yticklabels([r"$-\pi/4$", r"$-\pi/8$", r"$0$", r"$\pi/8$", r"$\pi/4$"])

    plt.savefig(
        "energy34.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
