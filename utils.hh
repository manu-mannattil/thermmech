#pragma once
#define _USE_MATH_DEFINES

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <valarray>
#include <vector>

std::random_device rand_dev;
std::mt19937_64 rand_gen(rand_dev());
std::uniform_real_distribution<double> unif(0, 1);
std::normal_distribution<double> normal(0);

// Generate a random uniform array of the given size with elements in [a, b].
std::valarray<double> uranarray(const size_t& size = 1, const double& a = -1.0,
                                const double& b = 1.0) {
    std::valarray<double> v(size);
    std::generate(begin(v), end(v), [a, b]() -> double { return a + (b - a) * unif(rand_gen); });
    return v;
}

// Generate a random normal array of the given size with mean m and std s.
std::valarray<double> nranarray(const size_t& size = 1, const double& m = 0,
                                const double& s = 1.0) {
    std::valarray<double> v(size);
    std::generate(begin(v), end(v), [m, s]() -> double { return m + s * normal(rand_gen); });
    return v;
}

// A class implementing the Metropolis algorithm.
template <typename T> class Metropolis {
  private:
    double dE;

  public:
    size_t acc, tot;
    double E;

    // The system is an object of some class with methods called energy(),
    // and cv().  beta is the inverse temperature.
    T sys;
    double beta;
    std::string move;
    double eps;
    bool center;
    Metropolis(const T& sys, const double& beta = 1.0, const std::string& move = "uniform",
               const double& eps = 1.0, const bool& center = false)
        : sys(sys), beta(beta), move(move), eps(eps), center(center) {
        E = sys.energy();
        acc = 0, tot = 0, dE = 0;
    }

    void step() {
        T new_sys = sys;
        if (move == "uniform")
            new_sys.q += uranarray(new_sys.q.size(), -eps, eps);
        else
            new_sys.q += nranarray(new_sys.q.size(), 0, eps);

        dE = new_sys.energy() - E;

        if (dE <= 0 || unif(rand_gen) <= std::exp(-beta * dE)) {
            sys = new_sys;

            // This isn't a center-of-mass centering of the mechanism.
            // This is just a mere shifting of the origin to avoid
            // overflow issues (which happen very rarely).  Don't use if
            // the CV isn't a shape coordinate!
            if (center)
                sys.q -= sys.q.sum() / sys.q.size();

            E += dE;
            acc += 1;
        }

        tot += 1;
    }

    std::vector<std::valarray<double>> sample(const long& n = 1000, const long& rate = 1,
                                              const float& print_every = 1) {
        std::vector<std::valarray<double>> result(n);
        int print_step = n * print_every / 100;

        for (auto i = 0; i < n; i++) {
            for (auto j = 0; j < rate; j++)
                step();

            result[i] = sys.cv();
            if (i % print_step == 0)
                std::cout << std::flush << 100 * i / n << "%; accept = " << stat() << "%\r";
        }
        std::cout << std::endl;

        return result;
    }

    float stat() const { return tot > 0 ? 100.0 * acc / tot : 0.0; }
};

// Compute the dot product of two valarrays in R^n.
inline double dot(const std::valarray<double>& a, const std::valarray<double>& b) {
    return (a * b).sum();
}

// Compute the cross product of two vectors in R^3.
std::valarray<double> cross(const std::valarray<double>& a, const std::valarray<double>& b) {
    std::valarray<double> v(3);
    v[0] = a[1] * b[2] - a[2] * b[1];
    v[1] = a[2] * b[0] - a[0] * b[2];
    v[2] = a[0] * b[1] - a[1] * b[0];
    return v;
}

// Computes the angle between the normals of two faces that share the
// edge e.  The edge e is assumed to be straddled by edges u and v.  The
// returned angle, which is ±180 ±(dihedral angle), is positive if e is
// in the direction of cross(u, v) and is negative otherwise.
double normangle(const std::valarray<double>& e, const std::valarray<double>& u,
                 const std::valarray<double>& v, const bool& dihedral = false) {
    int sign;
    if (dot(cross(u, v), e) >= 0)
        sign = 1;
    else
        sign = -1;

    std::valarray<double> n1(3), n2(3);
    n1 = cross(u, e), n2 = cross(e, v);
    double d = dot(n1, n2) / sqrt(dot(n1, n1) * dot(n2, n2));

    if (dihedral)
        return sign * 180 * (1 - acos(d) / M_PI);
    else
        return sign * 180 * acos(d) / M_PI;
}

// Result of the intersection test.
struct Result {
    bool r = false;
    std::valarray<double> T1;
    std::valarray<double> T2;
};

// Checks if the given triangles have an intersection [1].  The choice
// of the variable names is inspired by the original paper.
//
// - Intersections between coplanar triangles will not be detected.
//   For triangulated origami, this means that an intersection of two
//   faces that share an edge will not be caught.
//
// - Two non-intersecting triangles will be deemed as intersecting if
//   a triangle's side lies in the plane of the other triangle.  In
//   Monte Carlo simulations this shouldn't be much of an issue since
//   such occurrences are probabilistically almost impossible.
//
// - Another case of degeneracy is when the triangles share a common
//   vertex.  To avoid counting this as an intersection, order the
//   vertices in A and B such that the first vertex in both correspond
//   to the common one.
//
// [1]: Tropp et al., Comp. Anim. Virtual Worlds 17, 527 (2006)
//      https://doi.org/10.1002/cav.115
Result triinter(const std::vector<std::valarray<double>>& A,
                const std::vector<std::valarray<double>>& B, const bool& points = true) {
    Result res;

    // Directed edges of triangle A.
    std::vector<std::valarray<double>> q = {A[1] - A[0], A[2] - A[0], A[2] - A[1]};

    // Directed edges of triangle B.
    //  P = B[0]
    std::vector<std::valarray<double>> p = {B[1] - B[0], B[2] - B[0]};

    // Vector r.  Note that r0 = r1, and r2 is only needed for computing
    // det(A(r2)), which is equal to det(A(r0)) - det(A(q0)).
    std::valarray<double> r = A[0] - B[0];

    // Stage 1 -------------------------------------------------------------

    // Minors of the determinant.
    std::array<double, 3> M = {p[0][1] * p[1][2] - p[0][2] * p[1][1],
                               p[0][0] * p[1][2] - p[0][2] * p[1][0],
                               p[0][0] * p[1][1] - p[0][1] * p[1][0]};

    // Determinants A(-q_i).
    std::array<double, 3> Aq = {-M[0] * q[0][0] + M[1] * q[0][1] - M[2] * q[0][2],
                                -M[0] * q[1][0] + M[1] * q[1][1] - M[2] * q[1][2], 0};
    Aq[2] = Aq[1] - Aq[0];

    // Determinants A(r_i).
    std::array<double, 3> Ar = {M[0] * r[0] - M[1] * r[1] + M[2] * r[2], 0, 0};
    Ar[1] = Ar[0];
    Ar[2] = Ar[0] - Aq[0];

    // Find legal beta indices.  We want to avoid two betas with a value
    // of zero.  This is to ensure that the case where A and B share
    // a common vertex, but don't have any other intersection points is
    // not considered as an intersection.
    int i = -1, j = -1;
    bool nozero = true;
    for (auto k = 0; k < 3; k++)
        if (Aq[k] != 0 and Ar[k] * Aq[k] >= 0 and Ar[k] * Aq[k] <= Aq[k] * Aq[k]) {
            if (i == -1)
                i = k;
            else if (Ar[k] != 0 or (Ar[k] == 0 and nozero)) {
                j = k;
                break;
            }
            if (Ar[k] == 0)
                nozero = false;
        }

    // Stage 2 -------------------------------------------------------------

    // Bail out if there aren't at least two legal betas.
    if (i == -1 or j == -1)
        return res;

    // Stage 3 -------------------------------------------------------------

    // Find one point of intersection T and the line segment t.
    std::array<int, 3> ind = {0, 0, 1};
    std::valarray<double> T = A[ind[i]] + Ar[i] / Aq[i] * q[i];
    std::valarray<double> t = A[ind[j]] + Ar[j] / Aq[j] * q[j] - T;

    // Stage 4 -------------------------------------------------------------

    std::valarray<double> R = T - B[0]; // for i = 1, 2.
    std::valarray<double> S = R - p[0]; // for i = 3.

    // Common determinants for both the gammas and deltas that appear in the
    // denominator.  If there is a common vertex between the two triangles, and
    // that vertex is chosen as Q0, then there won't be any valid delta since
    // all Ci will be zero (as t = 0).
    std::array<double, 3> C = {-t[0] * p[0][1] + t[1] * p[0][0], -t[0] * p[1][1] + t[1] * p[1][0],
                               0};
    C[2] = C[1] - C[0];

    // Determinants for the deltas.  Note that we only use the first two
    // coordinates of S and R since the system of equations is overdetermined.
    std::array<double, 3> D = {-t[0] * R[1] + t[1] * R[0], 0, -t[0] * S[1] + t[1] * S[0]};
    D[1] = D[0];

    // Determinants for the gammas.
    std::array<double, 3> G = {R[0] * p[0][1] - R[1] * p[0][0], R[0] * p[1][1] - R[1] * p[1][0],
                               S[0] * (p[1][1] - p[0][1]) - S[1] * (p[1][0] - p[0][0])};

    // Find legal delta indices.  Again, avoid the case where there are
    // two zeros.
    i = -1, j = -1;
    nozero = true;
    for (auto k = 0; k < 3; k++)
        if (C[k] != 0 and D[k] * C[k] >= 0 and D[k] * C[k] <= C[k] * C[k]) {
            if (i == -1)
                i = k;
            else if (D[k] != 0 or (D[k] == 0 and nozero)) {
                j = k;
                break;
            }
            if (D[k] == 0)
                nozero = false;
        }

    // Stage 5 -------------------------------------------------------------

    // Bail out if there aren't at least two legal deltas.
    if (i == -1 or j == -1)
        return res;

    // The number of legal gammas tell us about the kind of
    // intersection.  'valid' is one of the valid indices.
    int legal = 0, valid = 0;
    for (auto k : {i, j})
        if (G[k] * C[k] >= 0 and G[k] * C[k] <= C[k] * C[k]) {
            legal += 1;
            valid = k;
        }

    // - If there are no legal gammas, but the gammas have opposite
    //   signs, then it's a Case II intersection with the penetrating
    //   smaller triangle being A and the bigger triangle being B.
    //
    // - If there is only one legal gamma, then it's a Case I intersection.
    //
    // - If there are two legal gammas, then it's a Case II intersection
    //   with the bigger triangle being A and the smaller triangle (which
    //   penetrates A) being B.
    if (legal > 0 or G[i] * C[i] * G[j] * C[j] <= 0)
        res.r = true;

    // Stage 6 -------------------------------------------------------------

    if (res.r and points) {
        // Function to compute an intersection point that falls on the edge of B.
        auto X = [&](int i) -> std::valarray<double> {
            if (i == 2)
                return B[0] + p[0] + D[i] / C[i] * (p[1] - p[0]);
            else
                return B[0] + D[i] / C[i] * p[i];
        };

        switch (legal) {
        case 0:
            res.T1 = T, res.T2 = T + t;
            break;
        case 1:
            // Even though X(i) which lies on the edge of B, is always
            // a point of intersection, only one of T and T + t is a point
            // of intersection.  To find out which one is the actual point
            // of intersection, see which one lies inside B.  First check if
            // T + t lies inside B by computing the alphas from known
            // determinants: alpha_0, alpha_1 = (G[1] - C[1]) / M[2], (C[0] - G[0]) / M[2]
            // If that fails, T is the intersection point.
            if ((G[1] - C[1]) * M[2] >= 0 and (C[0] - G[0]) * M[2] >= 0 and
                (G[1] - G[0] + C[0] - C[1]) * M[2] <= M[2] * M[2])
                res.T1 = X(valid), res.T2 = T + t;
            else
                res.T1 = X(valid), res.T2 = T;
            break;
        default:
            // Since A is the bigger triangle, both the intersection points
            // lie on the edges of B.
            res.T1 = X(i), res.T2 = X(j);
        }
    }
    return res;
}
