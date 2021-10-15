# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from subprocess import run

bluesmoke = LinearSegmentedColormap.from_list(
    "bluesmoke",
    [[8 / 15, 10 / 15, 12 / 15], [11 / 15, 12 / 15, 13 / 15], [14 / 15, 14 / 15, 14 / 15]]
)

class TorusAndBranches:
    def __init__(
        self, R=2.0, r=0.70, asp=2.0, points=500, elev=30, azim=20, aspect=(1, 1, 1), axlim=1.6
    ):
        self.R = R
        self.r = r
        self.asp = asp
        self.fig = plt.figure()
        self.points = points
        self.ax = self.fig.add_subplot(projection="3d", proj_type="ortho")
        self.ax.set_axis_off()
        self.ax.view_init(elev, azim)
        self.ax.set_box_aspect(aspect)
        self.ax.set_xlim((-axlim, axlim))
        self.ax.set_ylim((-axlim, axlim))
        self.ax.set_zlim((-axlim, axlim))

    def angle_to_torus(self, u, v):
        x = (self.R + self.r * np.cos(u)) * np.cos(v)
        y = (self.R + self.r * np.cos(u)) * np.sin(v)
        z = self.r * np.sin(u)
        return x, y, z

    def add_torus(self, *args, **kwargs):
        u = np.linspace(0, 2 * np.pi, self.points)
        v = np.linspace(0, 2 * np.pi, self.points)
        u, v = np.meshgrid(u, v)
        x, y, z = self.angle_to_torus(u, v)
        return self.ax.plot_surface(x, y, z, antialiased=True, linewidth=0, *args, **kwargs)

    def add_branch(self, branch=1, t1=0, t2=2 * np.pi, *args, **kwargs):
        u = np.linspace(t1, t2, self.points)
        if branch == 1:
            # Parallel branch.
            cos_v, sin_v = np.cos(u), np.sin(u)
        else:
            # Twisted branch.
            cos_v = ((1 + self.asp**2) * np.cos(u) -
                     2 * self.asp) / (1 + self.asp**2 - 2 * self.asp * np.cos(u))
            sin_v = (1 - self.asp**2) * np.sin(u) / (1 + self.asp**2 - 2 * self.asp * np.cos(u))

        x = (self.R + self.r * np.cos(u)) * cos_v
        y = (self.R + self.r * np.cos(u)) * sin_v
        z = self.r * np.sin(u)
        return self.ax.plot(x, y, z, *args, **kwargs)

    def add_point(self, t1, t2, *args, **kwargs):
        x, y, z = self.angle_to_torus(t1, t2)
        return self.ax.plot([x, x], [y, y], [z, z], *args, **kwargs)

    def render(self):
        canvas = FigureCanvas(self.fig)
        string, (width, height) = canvas.print_to_buffer()
        return np.frombuffer(string, np.uint8).reshape((height, width, 4))

def add_linkage(axis, branch=1, u=0, a=0, b=0, asp=2.0, short=0.12, gap=0.015):
    if branch == 1:
        axis.plot([a, a + asp*short], [b, b], "C0")
        axis.plot([a, a + short * np.cos(u)], [b, b + short * np.sin(u)],
                  "C0o-",
                  markerfacecolor="w",
                  markeredgecolor="k",
                  markeredgewidth=3,
                  zorder=5)
        axis.plot([a + short * np.cos(u), a + short * (np.cos(u) + asp)],
                  [b + short * np.sin(u), b + short * np.sin(u)], "C0")
        axis.plot([a + asp*short, a + short * (np.cos(u) + asp)], [b, b + short * np.sin(u)],
                  "C0o-",
                  markerfacecolor="w",
                  markeredgecolor="k",
                  markeredgewidth=3,
                  zorder=5)
    else:
        cos_v = ((1 + asp**2) * np.cos(u) - 2*asp) / (1 + asp**2 - 2 * asp * np.cos(u))
        sin_v = (1 - asp**2) * np.sin(u) / (1 + asp**2 - 2 * asp * np.cos(u))

        # Point where the bars appear to cross.
        cross = short * (np.cos(u) * sin_v - (asp+cos_v) * np.sin(u)) / (sin_v - np.sin(u))

        axis.plot([a, a + cross - gap], [b, b], "C3")
        axis.plot([a + cross + gap, a + asp*short], [b, b], "C3")
        axis.plot([a, a + short * np.cos(u)], [b, b + short * np.sin(u)],
                  "C3o-",
                  markerfacecolor="w",
                  markeredgecolor="k",
                  markeredgewidth=3,
                  zorder=5)
        axis.plot([a + asp*short, a + short * (cos_v+asp)], [b, b + short*sin_v],
                  "C3o-",
                  markerfacecolor="w",
                  markeredgecolor="k",
                  markeredgewidth=3,
                  zorder=5)
        axis.plot([a + short * np.cos(u), a + short * (cos_v+asp)],
                  [b + short * np.sin(u), b + short*sin_v], "C3")

rc = {
    "lines.linewidth": 4,
    "lines.markersize": 17,
    "figure.figsize": [16, 9],
    "figure.dpi": 80,
    "savefig.dpi": 80,
}

with plt.rc_context(rc):
    tab = TorusAndBranches()
    tab.add_torus(rstride=5, cstride=5, cmap=bluesmoke, alpha=0.65, zorder=5)
    tab.add_branch(t1=-0.2 * np.pi, t2=1.25 * np.pi, color="C0", zorder=5)
    tab.add_branch(t1=-0.2 * np.pi, t2=-0.75 * np.pi, color="C0", zorder=0)
    tab.add_branch(branch=2, t1=-0.2 * np.pi, t2=1.3 * np.pi, color="C3", zorder=5)
    tab.add_branch(branch=2, t1=-0.2 * np.pi, t2=-0.7 * np.pi, color="C3", zorder=0)

    tab.add_point(np.pi, np.pi, "ko", zorder=20)
    tab.add_point(0, 0, "ko", zorder=20)

    asp = 2.0
    for i in range(250):
        u = i * 2 * np.pi / (250-1)
        v = np.arctan2((1 - asp**2) * np.sin(u), ((1 + asp**2) * np.cos(u) - 2*asp))
        if u < 1.25 * np.pi or u > 1.8 * np.pi:
            p1, = tab.add_point(u, u, "C0o", zorder=10)
        else:
            p1, = tab.add_point(u, u, "C0o", zorder=0)
        if u < 1.3 * np.pi or u > 1.8 * np.pi:
            p2, = tab.add_point(u, v, "C3o", zorder=10)
        else:
            p2, = tab.add_point(u, v, "C3o", zorder=0)

        image = tab.render()
        p1.remove()
        p2.remove()

        fig, ax = plt.subplots()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 0.5625))
        ax.set_aspect("equal")

        add_linkage(ax, branch=1, u=u, a=0.60, b=0.15)
        add_linkage(ax, branch=2, u=u, a=0.60, b=0.4125)

        ax.set_axis_off()
        fig.figimage(image, xo=-320, yo=0, zorder=-1)
        fig.tight_layout()
        fig.savefig("anim_{:04d}.png".format(i))
        print("anim_{:04d}.png written".format(i))
        plt.close(fig)

    run(
        r"""ffmpeg
        -f image2
        -framerate 24
        -i anim_%04d.png
        -c:v libx264
        -preset veryslow
        -crf 20
        -pix_fmt yuv420p
        4bar.mp4""".split()
    )

    # By default, the GIFs produced by ffmpeg use a small color palette,
    # so we need to increase it: https://superuser.com/a/556031
    run(
        r"""ffmpeg
        -f image2
        -i anim_%04d.png
        -vf scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse
        -loop 0
        4bar.gif""".split()
    )
