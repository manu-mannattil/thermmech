# -*- coding: utf-8 -*-
"""Fuzz testing for the triangle-triangle intersection test."""

import numpy as np
from utils import triinter
from numpy.testing import assert_allclose, run_module_suite

# Number of random triangles to generate in each case.
N = 10 * 1000

def plot(A, B):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")

    ax.plot([A[0][0], A[1][0]], [A[0][1], A[1][1]], [A[0][2], A[1][2]], "C0")
    ax.plot([A[1][0], A[2][0]], [A[1][1], A[2][1]], [A[1][2], A[2][2]], "C0")
    ax.plot([A[2][0], A[0][0]], [A[2][1], A[0][1]], [A[2][2], A[0][2]], "C0")
    ax.plot([B[0][0], B[1][0]], [B[0][1], B[1][1]], [B[0][2], B[1][2]], "C1")
    ax.plot([B[1][0], B[2][0]], [B[1][1], B[2][1]], [B[1][2], B[2][2]], "C1")
    ax.plot([B[2][0], B[0][0]], [B[2][1], B[0][1]], [B[2][2], B[0][2]], "C1")

    plt.show()

def triinter_sort(A, B):
    """Consistently sort the intersection points for testing purposes."""
    result, T1, T2 = triinter(A, B)
    if result and T1[0] < T2[0]:
        return result, T1, T2
    else:
        return result, T2, T1

def triinter_cpp(A, B):
    """C++ version of triinter."""
    stod = lambda s: None if s == "nan" else float(s)

def make_inter(case=1):
    """Make two random triangles in R^3 that intersect at known points."""
    # Corners of triangle B.
    P, O1, O2 = np.random.random(3), np.random.random(3), np.random.random(3)

    # Two edges of B.
    p1 = O1 - P
    p2 = O2 - P

    # An intersection point lying inside triangle B.
    alpha1, alpha2 = 0.5 * np.random.random(), 0.5 * np.random.random()
    T1 = P + alpha1*p1 + alpha2*p2

    if case == 1:
        # An intersection point lying on the edge of triangle B.
        alpha1, alpha2 = np.random.random(), 0
    else:
        # Another intersection point lying inside triangle B.
        alpha1, alpha2 = 0.5 * np.random.random(), 0.5 * np.random.random()

    # Second intersection point.
    T2 = P + alpha1*p1 + alpha2*p2

    n = np.cross(p1, p2)
    n /= np.linalg.norm(n)

    # Two corners of A.  The edge formed by these corners is perpendicular
    # to the plane of B.
    Q1 = T1 + 0.5*n
    Q3 = T1 - 0.5*n

    # Third corner of A.
    if case == 1:
        O3 = T1 + 2 * (T2-T1)
        O3 = Q1 + 2 * (O3-Q1)
    else:
        O3 = T2 + (T2-Q1)

    A, B = np.array([Q1, Q3, O3]), np.array([P, O1, O2])

    # More fuzzing.
    np.random.shuffle(A)
    np.random.shuffle(B)

    # Even more fuzzing.
    if np.random.randint(2) == 1:
        A, B = B, A

    # Sort the same way as triinter_sort() does.
    if T1[0] < T2[0]:
        return T1, T2, A, B
    else:
        return T2, T1, A, B

def make_inter_1p():
    """Make two random triangles in R^3 that intersect at "one" point."""
    # Corners of triangle B.
    b1, b2, b3 = np.random.random(3), np.random.random(3), np.random.random(3)

    # Two edges of B.
    p1 = b2 - b1
    p2 = b3 - b1

    # An intersection point lying inside triangle B, which is also a corner of A.
    alpha1, alpha2 = 0.5 * np.random.random(), 0.5 * np.random.random()
    a1 = b1 + alpha1*p1 + alpha2*p2

    n = np.cross(p1, p2)
    n /= np.linalg.norm(n)

    # a1 has to be moved slightly downwards.  If a1 is exactly on the
    # plane of the triangle, the algorithm fails (which is expected).
    a1 -= 1e-10 * n
    a2 = b2 + (0.1 + np.random.random()) * n
    a3 = b3 + (0.1 + np.random.random()) * n

    A, B = np.array([a1, a2, a3]), np.array([b1, b2, b3])

    # More fuzzing.
    np.random.shuffle(A)
    np.random.shuffle(B)
    if np.random.randint(2) == 1:
        A, B = B, A

    return a1, A, B

def make_inter_common():
    """Make two intersecting triangles in R^3 that share a common vertex."""
    # Corners of triangle B.
    b1, b2, b3 = np.random.random(3), np.random.random(3), np.random.random(3)

    # Two edges of B.
    p1 = b2 - b1
    p2 = b3 - b1

    # An intersection point lying inside triangle B, which is also a corner of A.
    alpha1, alpha2 = 0.5 * np.random.random(), 0.5 * np.random.random()
    a2 = b1 + alpha1*p1 + alpha2*p2

    n = np.cross(p1, p2)
    n /= np.linalg.norm(n)

    # Third vertex of A is not in B's plane.
    a3 = b3 + (0.1 + np.random.random()) * n

    # a2 has to be moved slightly downwards.  If a2 is exactly on the
    # plane of the triangle, the algorithm fails (which is expected).
    a2 -= 1e-10 * n

    # Fuzz by interchanging the 2nd and 3rd vertex of A.
    # First vertex has to be the common point.
    _a2, _a3 = a2, a3
    if np.random.randint(2) == 1:
        _a2, _a3 = a3, a2
    A, B = np.array([b1, _a2, _a3]), np.array([b1, b2, b3])

    # More fuzzing.
    if np.random.randint(2) == 1:
        A, B = B, A

    # Sort the same way as triinter_sort() does.
    if b1[0] < a2[0]:
        return b1, a2, A, B
    else:
        return a2, b1, A, B

def make_nointer():
    """Make two random non-intersecting triangles in R^3."""
    # Corners of triangle B.
    b1, b2, b3 = np.random.random(3), np.random.random(3), np.random.random(3)

    # Two edges of B.
    p1 = b2 - b1
    p2 = b3 - b1

    n = np.cross(p1, p2)
    n /= np.linalg.norm(n)

    a1 = b1 + (0.1 + np.random.random()) * n
    a2 = b2 + (0.1 + np.random.random()) * n
    a3 = b3 + (0.1 + np.random.random()) * n

    A, B = np.array([a1, a2, a3]), np.array([b1, b2, b3])

    # More fuzzing.
    np.random.shuffle(A)
    np.random.shuffle(B)
    if np.random.randint(2) == 1:
        A, B = B, A

    return A, B

def make_nointer_beta():
    """Make two random non-intersecting triangles in R^3 that pass the beta test."""
    # Corners of triangle B.
    b1, b2, b3 = np.random.random(3), np.random.random(3), np.random.random(3)

    # Two edges of B.
    p1 = b2 - b1
    p2 = b3 - b1

    n = np.cross(p1, p2)
    n /= np.linalg.norm(n)

    T = b1 + (0.5 + 0.5 * np.random.random()) * p1 + (0.5 + 0.5 * np.random.random()) * p2
    a1 = T + np.random.random() * n
    a2 = T - np.random.random() * n
    a3 = b1 + (1.5 + 0.5 * np.random.random()) * p1 + (1.5 + 0.5 * np.random.random()) * p2

    A, B = np.array([a1, a2, a3]), np.array([b1, b2, b3])

    # More fuzzing.
    if np.random.randint(2) == 1:
        A, B = B, A

    return A, B

def make_nointer_common():
    """Make two non-intersecting triangles in R^3 that share a common vertex."""
    # Corners of triangle B.
    b1, b2, b3 = np.random.random(3), np.random.random(3), np.random.random(3)

    # Two edges of B.
    p1 = b2 - b1
    p2 = b3 - b1

    n = np.cross(p1, p2)
    n /= np.linalg.norm(n)

    a2 = b2 + (0.1 + np.random.random()) * n
    a3 = b3 + (0.1 + np.random.random()) * n

    A, B = np.array([b1, a2, a3]), np.array([b1, b2, b3])

    # More fuzzing.
    if np.random.randint(2) == 1:
        A, B = B, A

    return A, B

def test_inter_case1():
    for i in range(N):
        T1, T2, A, B = make_inter(case=1)
        result, X1, X2 = triinter_sort(A, B)
        assert result == True
        assert_allclose(T1, X1)
        assert_allclose(T2, X2)

def test_inter_case2():
    for i in range(N):
        T1, T2, A, B = make_inter(case=2)
        result, X1, X2 = triinter_sort(A, B)
        assert result == True
        assert_allclose(T1, X1)
        assert_allclose(T2, X2)

def test_inter_1p():
    for i in range(N):
        T, A, B = make_inter_1p()
        result, X1, X2 = triinter(A, B)
        assert result == True
        assert_allclose(T, X1, atol=1e-5, rtol=1e-5)
        assert_allclose(T, X2, atol=1e-5, rtol=1e-5)

def test_inter_common():
    for i in range(N):
        T1, T2, A, B = make_inter_common()
        result, X1, X2 = triinter_sort(A, B)
        assert result == True
        assert_allclose(T1, X1, atol=1e-5, rtol=1e-5)
        assert_allclose(T2, X2, atol=1e-5, rtol=1e-5)

def test_nointer():
    for i in range(N):
        A, B = make_nointer()
        assert triinter(A, B)[0] == False

def test_nointer_beta():
    for i in range(N):
        A, B = make_nointer_beta()
        assert triinter(A, B)[0] == False

def test_nointer_common():
    for i in range(N):
        A, B = make_nointer_common()
        assert triinter(A, B)[0] == False

if __name__ == "__main__":
    run_module_suite()
