#pragma once
#define _USE_MATH_DEFINES

#include <algorithm>
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

// Return a list of n recursive roots of the given factor.
std::valarray<double> nroot(int n, double factor = M_E) {
    std::valarray<double> v(n);
    for (auto i = 0; i < n; i++)
        v[i] = (factor = std::sqrt(factor));

    return v;
}

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
