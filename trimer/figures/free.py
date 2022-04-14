# -*- coding: utf-8 -*-
"""Free-energy landscape of the singular trimer."""

import numpy as np
import matplotlib.pyplot as plt
import mmmpl

rc = {
    "mmmpl.doc": "aps",
    "mmmpl.tex": True,
    "mmmpl.tex.font": "fourier",
    "figure.figsize": [3.06, 1.87]
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    ax.set_xlabel(r"$\theta$")
    ax.set_xlim((-np.pi, np.pi))
    ticks, labels = mmmpl.ticklabels(-np.pi, np.pi, 5, np.pi, r"\pi")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    ax.set_ylabel(r"$\Delta\mathcal{A}_{\hat{\theta}}(\theta)$")
    ax.set_ylim((-1.5, 0.25))

    # MC free energy.
    t1, a1 = np.loadtxt("../data/trimer_free.dat", unpack=True)
    ax.plot(t1, a1, "k", label="numerics")

    t1 = np.linspace(-np.pi, np.pi, 1000) 
    a1 = 7/4 * (np.log(1 + np.cos(t1) ** 2) - np.log(2))
    ax.plot(t1, a1, "C3--", label="theory")

    ax.legend(loc=(0.35, 0.15))

    plt.savefig(
        "trimer_free.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
