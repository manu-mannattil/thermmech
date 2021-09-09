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
    x1, y1 = np.loadtxt("free_numer.dat", unpack=True)
    ax.plot(x1, y1 - y1[int(len(y1) / 2)], "k", label="numerical")

    # Asymptotic (regular).
    x2, y2 = np.loadtxt("../free_theor.dat", unpack=True)
    y2 = y2 - y1[int(len(y1) / 2)]  # make plots coincide
    ax.plot(x2, y2, "C0--", label="asymptotic")

    # Asymptotic (singular).
    x3, y3 = np.loadtxt("../free_sing.dat", unpack=True)
    x3 = np.concatenate([-x3[::-1][:-1], x3])
    y3 = y3 - y3[0]
    y3 = np.concatenate([y3[::-1][:-1], y3])
    ax.plot(x3, y3, "C3--", label="singular")

    ax.set_ylim((-1, 3.75))
    ax.set_xlim((-180, 180))

    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\Delta\mathcal{A}_{\hat{\theta}_1}(\theta_1)$")

    ax.legend(loc=(0.02, 0.05))

    inset_top = fig.add_axes([0.342, 0.785, 0.34, 0.07])
    inset_top.spines["bottom"].set_visible(False)
    inset_top.tick_params(which="both", bottom=False, labelbottom=False)
    inset_top.tick_params(which="minor", left=False)
    for item in ([inset_top.title, inset_top.xaxis.label, inset_top.yaxis.label] +
                 inset_top.get_xticklabels() + inset_top.get_yticklabels()):
        item.set_fontsize(8.5)

    inset_top.tick_params(axis="both", which="major", pad=1)
    inset_top.set_xlim(-180 - 0.1, 180 + 0.1)
    inset_top.set_ylim(10, 35)
    inset_top.set_yticks([35])
    inset_top.set_yticklabels([r"$\infty$"])

    y2i = np.interp(x1, x2, y2)
    y1 = y1 - y1[int(len(y1) / 2)]
    inset_top.plot(x1, np.abs(y2i - y1), "C0")

    # Broken axis: https://stackoverflow.com/a/32186074
    d = 0.04  # width of //
    m = 2.5  # slope of //
    h = 0.2  # height of //
    kwargs = dict(transform=inset_top.transAxes, color="k", clip_on=False)
    inset_top.plot([-d, d], [-m * d, m * d], linewidth=0.5, **kwargs)
    inset_top.plot([-d, d], [-m * d - h, m*d - h], linewidth=0.5, **kwargs)
    inset_top.plot([1 - d, 1 + d], [-m * d, m * d], linewidth=0.5, **kwargs)
    inset_top.plot([1 - d, 1 + d], [-m * d - h, m*d - h], linewidth=0.5, **kwargs)

    inset_bot = fig.add_axes([0.342, 0.615, 0.34, 0.15])
    inset_bot.spines['top'].set_visible(False)
    for item in ([inset_bot.title, inset_bot.xaxis.label, inset_bot.yaxis.label] +
                 inset_bot.get_xticklabels() + inset_bot.get_yticklabels()):
        item.set_fontsize(7)

    inset_bot.tick_params(axis="both", which="major", pad=1.2)
    inset_bot.set_xlim(-180, 180)
    inset_bot.set_xlabel(r"$\theta_1$", labelpad=1)
    inset_bot.set_xticks([-180, 0, 180])
    inset_bot.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    inset_bot.set_ylim((0, 1.2))
    inset_bot.set_yticklabels([r"$0$", r"$1$"])

    inset_bot.plot(x1, np.abs(y2i - y1), "C0")  # regular
    y1i = np.interp(x3, x1, y1)
    inset_bot.plot(x3, np.abs(y3 - y1i), "C3")  # singular

    plt.savefig(
        "free.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
