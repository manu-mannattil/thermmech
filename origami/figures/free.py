# -*- coding: utf-8 -*-
"""Free-energy landscape of the origami."""

import numpy as np
import matplotlib.pyplot as plt
import charu
from origami import *

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "figure.figsize": [2.75, 1.7]
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    x1, y1 = np.loadtxt("../data/free_mc.dat", unpack=True) # numerical
    x2, y2 = np.loadtxt("../data/free_harmonic.dat", unpack=True) # asymptotic (harmonic)
    x3, y3 = np.loadtxt("../data/free_quartic_q4q7.dat", usecols=(0, 1), unpack=True) # asymptotic (quartic)

    # Plot MC free energy.
    ax.plot(x1, y1 - y1[int(len(y1) / 2)], "k", label="numerical")

    # The harmonic approximation is invalid at rho = 0, which is the
    # point that we use as the reference for both the quartic and MC
    # free energies.  This means that we should subtract off either the
    # quartic or MC free energies at rho = 0 from the harmonic free
    # energy.  For example, using the quartic free energy,
    # y2 = y2 - y3[0]

    # But we should also realize that the quartic free energy is always
    # slightly overestimated at rho = 0 due to the fact that the
    # numerical integration is approximate.  So it's better to use the
    # MC free energy as the reference.
    y2 = (y2 - y1[int(len(y1) / 2)]) - (y2[-1] - y1[-1])
    ax.plot(x2, y2, "C0--", label="harmonic")

    # Mirror the asymptotic free energy for rho < 0.
    x3 = np.concatenate([-x3[::-1][:-1], x3])
    y3 = y3 - y3[0]
    y3 = np.concatenate([y3[::-1][:-1], y3])
    ax.plot(x3, y3, "C3--", label="quartic")

    ax.set_xlabel(r"$\rho_1$")
    ax.set_xlim((-180, 180))
    ticks, labels = charu.ticklabels(-np.pi, np.pi, 5, np.pi, divstr=r"\pi")
    ax.set_xticks(ticks * 180 / np.pi)
    ax.set_xticklabels(labels)

    ax.set_ylabel(r"$\Delta\mathcal{A}_{\hat{\rho}_1}(\rho_1)$", labelpad=-2)
    ax.set_ylim((-1, 4))

    ax.legend(loc=(0.02, 0.02))

    inset = fig.add_axes([0.342, 0.62, 0.34, 0.22])
    for item in ([inset.title, inset.xaxis.label, inset.yaxis.label] +
                 inset.get_xticklabels() + inset.get_yticklabels()):
        item.set_fontsize(7)

    inset.tick_params(axis="both", which="major", pad=1.2)
    inset.set_xlabel(r"$\rho_1$", labelpad=1)
    inset.set_xlim(-180, 180)
    ticks, labels = charu.ticklabels(-np.pi, np.pi, 3, np.pi, divstr=r"\pi")
    inset.set_xticks(ticks * 180 / np.pi)
    inset.set_xticklabels(labels)
    inset.set_ylim((0, 1))
    inset.set_yticklabels([r"$0$", r"$1$"])

    y2i = np.interp(x1, x2, y2)
    y1 = y1 - y1[int(len(y1) / 2)]
    inset.plot(x1, np.abs(y2i - y1), "C0")  # harmonic
    y1i = np.interp(x3, x1, y1)
    inset.plot(x3, np.abs(y3 - y1i), "C3")  # quartic

    plt.savefig(
        "free.pdf",
        crop=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
