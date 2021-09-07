#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.special import pbdv
import matplotlib.pyplot as plt
import numpy as np
import mmmpl
from utils import *

beta = 10
xx = np.loadtxt("rotor_met.dat", usecols=(0, ))

rc = {
    "mmmpl.doc": "standard",
    "mmmpl.tex": True,
    "mmmpl.tex.font": "mathtime",
    "mmmpl.wide": True,
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()
    n = 200
    x = np.linspace(-1.5, 1.5, num=n)

    # Guide lines.
    ax.plot([-1, -1], [-1, 1], color="#cccccc")
    ax.plot([1, 1], [-1, 1], color="#cccccc")
    ax.plot([-2, 2], [0, 0], color="#cccccc")

    ax.set_ylim((-0.8, 0.2))
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim((-0.8, 0.2))
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\beta(\mathcal{A}_{\hat{x}}(x) - \mathcal{A}_{\hat{x}}(0))$")

    # Histogram
    hist = pdf(xx, x)
    ax.plot(x, -np.log(hist / hist[int(n / 2)]), "k")

    # KDE
    # from scipy import stats
    # kde = stats.gaussian_kde(xx)
    # ax.plot(x, -np.log(kde(x) / kde(0)), "C3", label="KDE")

    # Actual free energy.
    free = 0.5 * beta * (x**4 - 2 * x**2) - np.log(pbdv(-0.5, np.sqrt(2 * beta) * (x**2 - 1))[0]) + \
        np.log(pbdv(-0.5, -np.sqrt(2 * beta))[0])
    ax.plot(x, free, "C3--")

    # Approximate free energy.
    x = x[np.abs(x) < 1]
    ax.plot(x, np.log(np.sqrt(1 - x*x)), "C0--")

    plt.subplots_adjust(left=0, top=1, bottom=0, right=1)
    plt.tight_layout()
    plt.savefig(
        "rotor.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
