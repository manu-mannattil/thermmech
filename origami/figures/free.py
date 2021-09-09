# -*- coding: utf-8 -*-
"""Free-energy landscape of the origami."""

import numpy as np
import matplotlib.pyplot as plt
import mmmpl
from origami import *

rc = {
    "mmmpl.doc": "aps",
    "mmmpl.tex": True,
    "mmmpl.tex.font": "newtx",
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    # Numerical.
    x1, y1 = np.loadtxt("../free_numer_10000.dat", unpack=True)
    ax.plot(x1, y1 - y1[int(len(y1) / 2)], "k", label="numerical")

    # Asymptotic (harmonic).
    x2, y2 = np.loadtxt("../free_theor.dat", unpack=True)
    y2 = y2 - y1[int(len(y1) / 2)]  # make plots coincide
    ax.plot(x2, y2, "C0--", label="harmonic")

    # Asymptotic (quartic).
    x3, y3 = np.loadtxt("../free_sing_10000.dat", unpack=True)
    x3 = np.concatenate([-x3[::-1][:-1], x3])
    y3 = y3 - y3[0]
    y3 = np.concatenate([y3[::-1][:-1], y3])
    ax.plot(x3, y3, "C3--", label="quartic")

    ax.set_xlabel(r"$\theta_1$")
    ax.set_xlim((-180, 180))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xticks([-180, -90, 0, 90, 180])

    ax.set_ylabel(r"$\Delta\mathcal{A}_{\hat{\theta}_1}(\theta_1)$")
    ax.set_ylim((-1, 4))

    ax.legend(loc=(0.02, 0.05))

    inset = fig.add_axes([0.342, 0.62, 0.34, 0.22])
    for item in ([inset.title, inset.xaxis.label, inset.yaxis.label] +
                 inset.get_xticklabels() + inset.get_yticklabels()):
        item.set_fontsize(7)

    inset.tick_params(axis="both", which="major", pad=1.2)
    inset.set_xlabel(r"$\theta_1$", labelpad=1)
    inset.set_xlim(-180, 180)
    inset.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    inset.set_xticks([-180, 0, 180])
    inset.set_ylim((0, 1))
    inset.set_yticklabels([r"$0$", r"$1$"])

    y2i = np.interp(x1, x2, y2)
    y1 = y1 - y1[int(len(y1) / 2)]
    inset.plot(x1, np.abs(y2i - y1), "C0")  # regular
    y1i = np.interp(x3, x1, y1)
    inset.plot(x3, np.abs(y3 - y1i), "C3")  # singular

    plt.savefig(
        "free.pdf",
        crop=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
