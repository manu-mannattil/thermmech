# -*- coding: utf-8 -*-
"""Common utility classes and functions."""

import numpy as np
from multiprocessing import Pool
from numba import njit
from scipy.linalg import null_space

RAD = np.pi / 180  # degree to radian
DEG = 180 / np.pi  # radian to degree

def pdf(a, bins):
    """Compute the probability density from a data set a and the given bins."""
    # Choose mid-point bins as the new bins.
    bins = np.sort(bins)
    new_bins = 0.5 * (bins[1:] + bins[:-1])

    # Insert first and last element of bins in new_bins.  This makes the first
    # and last bin half the size of all the other bins (if uniformly-sized bins
    # are chosen).
    new_bins = np.insert(new_bins, 0, bins[0])
    new_bins = np.append(new_bins, bins[-1])

    # Divide by the bin width.  This makes it possible to add the PDF of
    # multiple data sets together and then normalize later.  Note that this is
    # different from the density option of np.histogram(), which normalizes the
    # histogram immediately.
    return np.histogram(a, new_bins, density=False)[0] / np.diff(new_bins)

def parallel_map(func, values, args=tuple(), kwargs=dict(), processes=None):
    """Use Pool.apply_async() to get a parallel map().

    Uses Pool.apply_async() to provide a parallel version of map().
    Unlike Pool's map() which does not let you accept arguments and/or
    keyword arguments, this one does.

    Parameters
    ----------
    func : function
        This function will be applied on every element of values in
        parallel.
    values : array
        Input array.
    args : tuple, optional (default: ())
        Additional arguments for func.
    kwargs : dictionary, optional (default: {})
        Additional keyword arguments for func.
    processes : int, optional (default: None)
        Number of processes to run in parallel.  By default, the output
        of cpu_count() is used.

    Returns
    -------
    results : array
        Output after applying func on each element in values.
    """
    # True single core processing, in order to allow the function to be
    # executed in a calling script.
    if processes == 1:
        return np.asarray([func(value, *args, **kwargs) for value in values])

    pool = Pool(processes=processes)
    results = [pool.apply_async(func, (value, ) + args, kwargs) for value in values]

    pool.close()
    pool.join()

    return np.asarray([result.get() for result in results])

@njit(fastmath=True)
def normangle(e, u, v, dihedral=False):
    """Compute the angle betweens the normals of faces that share an edge.

    Computes the angle between the normals of two faces that share the edge e.
    The edge e is assumed to be straddled by edges u and v.  The returned
    angle, which is +/- 180 +/- (dihedral angle), is positive if e is in the
    direction of cross(u, v) and is negative otherwise.

    For an origami that lies flatly on the xy plane, choose e, u, and v such
    that the vectors u -> e -> v are in a anticlockwise arrangement around the
    positive z axis.  This means that a mountain fold (as perceived by looking
    down from the positive z axis) has a positive angle, e.g.,
    >>> round(normangle([1, 1, 1], [1, 0, 0], [0, 1, 0]), 5)
    60.0

    On the other hand, if the fold is a valley fold (from the vantage point of
    the positive z axis) then the returned angle is negative, e.g.,
    >>> round(normangle([1, 1, -1], [1, 0, 0], [0, 1, 0], dihedral=True), 5)
    -120.0
    """
    e, u, v, = np.asarray(e), np.asarray(u), np.asarray(v)

    # Choose the sign of the box product [e, u, v] as the sign of the angle.
    # We don't use np.sign() since the convention is np.sign(0) = 0.
    if np.cross(u, v).dot(e) >= 0:
        sign = 1
    else:
        sign = -1

    # Compute the normals of the faces with edges (e, u) and (e, v).
    n1, n2 = np.cross(u, e), np.cross(e, v)
    dot = n1.dot(n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))

    if dihedral:
        return sign * (180 - np.arccos(dot) * DEG)
    else:
        return sign * np.arccos(dot) * DEG

def plot_origami(
    coords,
    faces,
    axis,
    grid=False,
    alpha=0.9,
    hue=0.54,
    val=1.0,
    contrast=0.3,
    azdeg=315,
    altdeg=45,
    sat_min=0.4,
    sat_max=1.0,
    scale=0.25,
    edgecolor="k",
):
    """Plot an origami."""
    from matplotlib.colors import LightSource, hsv_to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    if not isinstance(axis, Axes3D):
        raise ValueError("{} must be an Axes3D instance".format(ax))

    if not grid:
        axis.set_axis_off()

    # Sane min and max.
    sat_min = max(0, sat_min)
    sat_max = min(sat_max, 1)
    sat_max = max(sat_min, sat_max)

    verts = np.array([coords[f] for f in faces])
    ls = LightSource(azdeg, altdeg)

    normals = []
    for v in verts:
        # Find the edges and the normal vector to each faces.
        e1, e2 = v[1] - v[0], v[2] - v[0]
        n = np.cross(e1, e2)
        normals.append(n / np.linalg.norm(n))

    # Compute the HSV colors for each face while making sure that the
    # saturation stays in the interval [sat_min, sat_max].
    normals = np.array(normals)
    sat = ls.shade_normals(normals, fraction=contrast)
    sat = (sat_max-sat_min) * sat + sat_min
    hue = hue * np.ones(sat.size)
    val = val * np.ones(sat.size)
    hsv = np.vstack([hue, val, sat]).T
    rgb = np.apply_along_axis(hsv_to_rgb, 1, hsv)

    axis.set_xlim(np.min(coords[:, 0]), np.max(coords[:, 0]))
    axis.set_ylim(np.min(coords[:, 1]), np.max(coords[:, 1]))
    axis.set_zlim(np.min(coords[:, 2]), np.max(coords[:, 2]))
    axis.add_collection3d(Poly3DCollection(verts, facecolors=rgb, alpha=alpha, edgecolor=edgecolor, sizes=(scale,)))

    return axis

class BranchParam:
    """Parameterize one-dimensional manifolds (i.e., branches)."""
    def __init__(self, f, jac, max_err=1e-12, max_iter=50):
        self.f = f
        self.jac = jac
        self.max_err = max_err
        self.max_iter = max_iter

    def tangent(self, q, qdot):
        ns = null_space(self.jac(q))
        # For regular points, the null space is one-dimensional.
        # However, the sign of the tangent changes.  Hence, to preserve
        # smoothness, once a new tagent is found, multiply it with the
        # sign of the dot product of the newly found tangent and the
        # previous one.
        if ns.shape[1] == 1:
            ns = ns.flatten()
            qdot = ns * np.sign(np.dot(qdot, ns))

        # If the nullspace is not one-dimensional, it means that we're
        # at a singularity or very close to one (at least by Python's
        # tolerance levels).  Thus, take the new tangent vector to be
        # the projection of the previous tangent vector onto the new
        # nullspace.  This is obviously not quite correct, since we
        # won't get the tangent to the actual branch in this case.  In
        # my tests, this got evaluated __only__ at the singularities.
        # Thus, the error is bound to be __very__ small.
        else:
            proj = ns.T.dot(qdot)
            qdot = ns.dot(proj)

        return qdot

    @staticmethod
    def interpolate(new_x, x, yy):
        """Interpolate an N-D array, which is parameterized by a 1-D array."""
        i = np.argsort(x)
        return np.array([np.interp(new_x, x[i], y[i]) for y in yy.T]).T

    def minimize(self, q):
        # Use the Gauss-Newton method to solve f(q) == 0.  I could have used
        # more sophisticated methods from scipy.optimize, but I want to keep
        # things simple.
        for i in range(self.max_iter):
            b = self.f(q)
            if np.linalg.norm(b) < self.max_err:
                return q
            elif i < self.max_iter - 1:
                q = q - np.linalg.lstsq(self.jac(q), b, rcond=None)[0]
            else:
                raise RuntimeError("Gauss-Newton solver exceeded 'max_iter'.")

    def arcpar(self, q0, qdot0, max_steps=100, step_length=1e-3):
        """Arc-length parameterization starting at q = q0, dq/ds = qdot0."""
        q, qdot = q0, qdot0
        qq = [q0]
        for i in range(max_steps):
            qdot = self.tangent(q, qdot)
            q = self.minimize(q + step_length*qdot)
            qq.append(q)

        return np.array(qq)

    def funpar(self, q0, qdot0, par, points=None, arc_steps=100, step_length=1e-3, passes=3):
        """Parameterize using the value of a function par()."""
        # Start with a set of points parameterized by the arc length.
        qq = self.arcpar(q0, qdot0, arc_steps, step_length)

        # Find the value of the parameter at each point.
        pp = np.apply_along_axis(par, 1, qq)

        if points is None:
            points = np.linspace(pp.min(), pp.max(), int((pp.max() - pp.min()) / step_length))
        else:
            points = np.asarray(points)
            if points.min() < pp.min() or points.max() > pp.max():
                raise ValueError(
                    "Chosen point range [{}, {}] not inside parameter range [{}, {}] on arc.".
                    format(points.min(), points.max(), pp.min(), pp.max())
                )

        for i in range(passes):
            # Interpolate within the points to form a list of points equispaced
            # in the parameter space.
            qq = self.interpolate(points, pp, qq)

            # Because of the interpolation, we've been kicked off the branch.
            # Project back to the branch again.
            for i, q in enumerate(qq):
                qq[i] = self.minimize(q)

            # Remove (duplicate) points with the same parameter value.
            pp = np.apply_along_axis(par, 1, qq)
            pp, i = np.unique(pp, return_index=True)
            qq = qq[i]

        return pp, qq
