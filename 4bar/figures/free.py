# -*- coding: utf-8 -*-
"""Free-energy landscape of the four-bar linkage."""

import numpy as np
import matplotlib.pyplot as plt
import mmmpl
from scipy.special import pbdv

# Parameters
beta = 10 * 1000
kappa = 1
lamb = 2
a = 1

# Nondimensional constants.
X = np.sqrt(beta * kappa) * lamb * a / (8 * np.abs(lamb - 1))
Y = np.abs((lamb-1) / (lamb+1))

t, a = np.loadtxt("free.dat", unpack=True)

rc = {
    "mmmpl.doc": "aps",
    "mmmpl.tex": True,
    "mmmpl.tex.font": "newtx",
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    ax.set_ylim((-0.5, 2))
    ax.set_xlim((-np.pi, np.pi))
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\Delta\mathcal{A}_{\hat{\theta}_1}(\theta_1)$", labelpad=0)

    # MC free energy.
    ax.plot(t, a, "k", label="numerical")

    # Asymptotic (regular).
    r = np.log(np.sqrt(X) * pbdv(-1 / 2, 0)[0] * np.abs(np.sin(t)))
    ax.plot(t, r, "C0--", label="asymptotic")

    # Asymptotic (|t| -> 0).
    t1 = t[(t > -8 * np.pi / 25) & (t < 8 * np.pi / 25)]
    a1 = a[(t > -8 * np.pi / 25) & (t < 8 * np.pi / 25)]
    s1 = X**2 * t1**4 - np.log(pbdv(-1 / 2, -2 * X * t1**2)[0] / pbdv(-1 / 2, 0)[0])
    ax.plot(t1, s1, "C3--", label="singular")

    # Asymptotic (t -> -pi).
    t2 = t[(t > -np.pi) & (t < -17 * np.pi / 25)]
    a2 = a[(t > -np.pi) & (t < -17 * np.pi / 25)]
    s2 = X**2 * Y**2 * (np.pi - np.abs(t2))**4 - np.log(
        np.sqrt(Y) * pbdv(-1 / 2, -2 * X * Y * (np.pi - np.abs(t2))**2)[0] / pbdv(-1 / 2, 0)[0]
    )
    ax.plot(t2, s2, "C3--")

    # Asymptotic (t -> pi).
    t3 = t[(t < np.pi) & (t > 17 * np.pi / 25)]
    a3 = a[(t < np.pi) & (t > 17 * np.pi / 25)]
    s3 = X**2 * Y**2 * (np.pi - t3)**4 - np.log(
        np.sqrt(Y) * pbdv(-1 / 2, -2 * X * Y * (np.pi - t3)**2)[0] / pbdv(-1 / 2, 0)[0]
    )
    ax.plot(t3, s3, "C3--")

    ax.legend(loc=(0.09, 0.12))

    inset_top = fig.add_axes([0.605, 0.41, 0.24, 0.05])
    inset_top.spines["bottom"].set_visible(False)
    inset_top.tick_params(which="both", bottom=False, labelbottom=False)
    inset_top.tick_params(which="minor", left=False)
    for item in ([inset_top.title, inset_top.xaxis.label, inset_top.yaxis.label] +
                 inset_top.get_xticklabels() + inset_top.get_yticklabels()):
        item.set_fontsize(8.5)

    inset_top.tick_params(axis="both", which="major", pad=1)
    inset_top.set_xlim(-np.pi - 0.1, np.pi + 0.1)
    inset_top.set_ylim(10, 50)
    inset_top.set_yticks([50])
    inset_top.set_yticklabels([r"$\infty$"])

    inset_top.plot(t, np.abs(1 - r/a), "C0")

    # Broken axis: https://stackoverflow.com/a/32186074
    d = 0.04 # width of //
    m = 2.5  # slope of //
    h = 0.2  # height of //
    kwargs = dict(transform=inset_top.transAxes, color="k", clip_on=False)
    inset_top.plot([-d, d], [-m * d, m * d], linewidth=0.5, **kwargs)
    inset_top.plot([-d, d], [-m * d - h, m*d - h], linewidth=0.5, **kwargs)
    inset_top.plot([1 - d, 1 + d], [-m * d, m * d], linewidth=0.5, **kwargs)
    inset_top.plot([1 - d, 1 + d], [-m * d - h, m*d - h], linewidth=0.5, **kwargs)

    inset_bot = fig.add_axes([0.605, 0.25, 0.24, 0.15])
    inset_bot.spines['top'].set_visible(False)
    for item in ([inset_bot.title, inset_bot.xaxis.label, inset_bot.yaxis.label] +
                 inset_bot.get_xticklabels() + inset_bot.get_yticklabels()):
        item.set_fontsize(7)

    inset_bot.tick_params(axis="both", which="major", pad=1)
    inset_bot.set_xlabel(r"$\theta_1$", labelpad=1)
    inset_bot.set_xlim(-np.pi - 0.1, np.pi + 0.1)
    inset_bot.set_xticks([-np.pi, 0, np.pi])
    inset_bot.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    inset_bot.set_ylim(0, 1.2)
    inset_bot.set_yticklabels([r"$0$", r"$1$"])

    inset_bot.plot(t, np.abs(r - a), "C0")

    # It doesn't make sense to plot the relative error near the singularities
    # as the difference is bound to be small.  Also, as t -> 0, most
    # definitions of relative errors give you a wildly oscillating result since
    # the denominator becomes small.
    inset_bot.plot(t1, np.abs(s1 - a1), "C3")
    inset_bot.plot(t2, np.abs(s2 - a2), "C3")
    inset_bot.plot(t3, np.abs(s3 - a3), "C3")

    plt.savefig(
        "4bar_free.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
