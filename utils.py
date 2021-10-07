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
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    axis.add_collection3d(
        Poly3DCollection(verts, facecolors=rgb, alpha=alpha, edgecolor=edgecolor, sizes=(scale, ))
    )

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

def triinter(A, B, points=True):
    """Check if the given triangles have an intersection.

    Checks if the given triangles have an intersection [1].  The choice
    of the variable names is inspired by the original paper.

    Parameters
    ----------
    A : array_like
        Three vertices of triangle A.
    B : array_like
        Three vertices of triangle B.
    points : bool, optional (default = True)
        Return the intersection point.

    Returns
    -------
    intersect : bool
        True if the two triangles have an intersection.
    T1 : ndarray or None
        First intersection point.
    T2 : ndarray or None
        Second intersection point.

    Notes
    -----

    - Intersections between coplanar triangles will not be detected.
      For triangulated origami, this means that an intersection of two
      faces that share an edge will not be caught.

    - Two non-intersecting triangles will be deemed as intersecting if
      a triangle's side lies in the plane of the other triangle.  In
      Monte Carlo simulations this shouldn't be much of an issue since
      such occurrences are probabilistically almost impossible.

    - Another case of degeneracy is when the triangles share a common
      vertex.  To avoid counting this as an intersection, order the
      vertices in A and B such that the first vertex in both correspond
      to the common one.

    [1]: Tropp et al., Comp. Anim. Virtual Worlds 17, 527 (2006)
         https://doi.org/10.1002/cav.115
    """
    A, B = np.asarray(A), np.asarray(B)

    # Directed edges of triangle A.
    q = [A[1] - A[0], A[2] - A[0], A[2] - A[1]]

    # Directed edges of triangle B.
    #   P = B[0]
    p = [B[1] - B[0], B[2] - B[0]]

    # Vector r.  Note that r0 = r1, and r2 is only needed for computing
    # det(A(r2)), which is equal to det(A(r0)) - det(A(q0)).
    r = A[0] - B[0]

    # Stage 1 --------------------------------------------------------------

    # Minors of the determinant.
    M = [
        p[0][1] * p[1][2] - p[0][2] * p[1][1], p[0][0] * p[1][2] - p[0][2] * p[1][0],
        p[0][0] * p[1][1] - p[0][1] * p[1][0]
    ]

    # Determinants A(-q_i).
    Aq = [
        -M[0] * q[0][0] + M[1] * q[0][1] - M[2] * q[0][2],
        -M[0] * q[1][0] + M[1] * q[1][1] - M[2] * q[1][2], 0
    ]
    Aq[2] = Aq[1] - Aq[0]

    # Determinants A(r_i).
    Ar = [M[0] * r[0] - M[1] * r[1] + M[2] * r[2], 0, 0]
    Ar[1] = Ar[0]
    Ar[2] = Ar[0] - Aq[0]

    # Find legal beta indices.  We want to avoid two betas with a value
    # of zero.  This is to ensure that the case where A and B share
    # a common vertex, but don't have any other intersection points is
    # not considered as an intersection.
    i, j, nozero = None, None, True
    for k in range(3):
        if Aq[k] != 0 and Ar[k] * Aq[k] >= 0 and Ar[k] * Aq[k] <= Aq[k] * Aq[k]:
            if i is None:
                i = k
            elif Ar[k] != 0 or (Ar[k] == 0 and nozero):
                j = k
                break
            if Ar[k] == 0:
                nozero = False

    # Stage 2 --------------------------------------------------------------

    # Bail out if there aren't at least two legal betas.
    if i is None or j is None:
        return False, None, None

    # Stage 3 --------------------------------------------------------------

    # Find one point of intersection T and the line segment t.
    #   Q = [A[0], A[0], A[1]]
    #   Q[i] = A[ind[i]]
    #   beta[i] = Ar[i] / Aq[i]
    ind = [0, 0, 1]
    T = A[ind[i]] + Ar[i] / Aq[i] * q[i]
    t = A[ind[j]] + Ar[j] / Aq[j] * q[j] - T

    # Stage 4 --------------------------------------------------------------

    R = T - B[0]  # for i = 1, 2.
    S = R - p[0]  # for i = 3.

    # Common determinants for both the gammas and deltas that appear in the
    # denominator.  If there is a common vertex between the two triangles, and
    # that vertex is chosen a Q0, then there won't be any valid delta since all
    # Ci will be zero (as t = 0).
    C = [-t[0] * p[0][1] + t[1] * p[0][0], -t[0] * p[1][1] + t[1] * p[1][0], 0]
    C[2] = C[1] - C[0]

    # Determinants for the deltas.  Note that we only use the first two
    # coordinates of S and R since the system of equations is overdetermined.
    D = [-t[0] * R[1] + t[1] * R[0], 0, -t[0] * S[1] + t[1] * S[0]]
    D[1] = D[0]

    # Determinants for the gammas.
    G = [
        R[0] * p[0][1] - R[1] * p[0][0], R[0] * p[1][1] - R[1] * p[1][0],
        S[0] * (p[1][1] - p[0][1]) - S[1] * (p[1][0] - p[0][0])
    ]

    # Find legal delta indices.  Again, avoid the case where there are
    # two zeros.
    i, j, nozero = None, None, True
    for k in range(3):
        if C[k] != 0 and D[k] * C[k] >= 0 and D[k] * C[k] <= C[k] * C[k]:
            if i is None:
                i = k
            elif D[k] != 0 or (D[k] == 0 and nozero):
                j = k
                break
            if D[k] == 0:
                nozero = False

    # Stage 5 --------------------------------------------------------------

    # There has to be at least two deltas for an intersection.
    if i is None or j is None:
        return False, None, None

    res = False

    # The number of legal gammas tell us about the kind of
    # intersection.  'valid' is one of the valid indices.
    legal, valid = 0, None
    for k in (i, j):
        if G[k] * C[k] >= 0 and G[k] * C[k] <= C[k] * C[k]:
            legal += 1
            valid = k

    # - If there are no legal gammas, but the gammas have opposite
    #   signs, then it's a Case II intersection with the penetrating
    #   smaller triangle being A and the bigger triangle being B.
    #
    # - If there is only one legal gamma, then it's a Case I intersection.
    #
    # - If there are two legal gammas, then it's a Case II intersection
    #   with the bigger triangle being A and the smaller triangle (which
    #   penetrates A) being B.
    if legal > 0 or G[i] * C[i] * G[j] * C[j] <= 0:
        res = True

    # Stage 6 --------------------------------------------------------------

    if res and points:
        # Function to compute an intersection point that falls on the edge of B.
        X = lambda i: B[0] + p[0] + D[i] / C[i] * (p[1] - p[0]) if i == 2 else B[0] + D[i] / C[i] * p[i]

        if legal == 0:
            return res, T, T + t
        elif legal == 1:
            # Even though X(i) which lies on the edge of B, is always
            # a point of intersection, only one of T and T + t is a point
            # of intersection.  To find out which one is the actual point
            # of intersection, see which one lies inside B.  First check if
            # T + t lies inside B by computing the alphas from known
            # determinants: alpha_0, alpha_1 = (G[1] - C[1]) / M[2], (C[0] - G[0]) / M[2]
            # If that fails, T is the intersection point.
            if ((G[1] - C[1]) * M[2] >= 0 and (C[0] - G[0]) * M[2] >= 0
                and (G[1] - G[0] + C[0] - C[1]) * M[2] <= M[2] * M[2]):
                return res, X(valid), T + t
            else:
                return res, X(valid), T
        else:
            # Since A is the bigger triangle, both the intersection points
            # lie on the edges of B.
            return res, X(i), X(j)
    else:
        return res, None, None
