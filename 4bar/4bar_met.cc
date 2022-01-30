// 4bar_met.cc -- sample the four-bar shape space using Metropolis MC.
//
// This file can be compiled with com: https://is.gd/compileanything
//
// com: : ${CXX:=c++}
// com: : ${CXXFLAGS:=-std=c++11 -Wall -Wextra -Werror -Ofast -static}
// com: ${CXX} {} ${CXXFLAGS} -o {.}
//

#include "../utils.hh"
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

// Generalized four-bar linkage with sides a, b, c, and d.
// Vertices are arranged in clock-wise order from the "origin" vertex
// (different from paper).
class FourBar {
  private:
    double a2, b2, c2, d2;
    double com_x = 0.0, com_y = 0;
    inline double square(const double& x) const { return x * x; }

  public:
    double a, b, c, d;
    valarray<double> q;
    FourBar(const double& a = 1.0, const double& b = 2.0, const double& c = 1.0,
            const double& d = 2.0)
        : a(a), b(b), c(c), d(d) {
        a2 = square(a), b2 = square(b), c2 = square(c), d2 = square(d);
        q = {0, 0, a, 0, a + b, 0, a + b - c, 0};
    }

    FourBar(const double& a, const double& b, const double& c, const double& d,
            const valarray<double>& q)
        : FourBar{a, b, c, d} {
        this->q = q;
    }

    double energy() const {
        return 1 / 8.0 *
               (square(square(q[2] - q[0]) + square(q[3] - q[1]) - a2) / a2 +
                square(square(q[4] - q[2]) + square(q[5] - q[3]) - b2) / b2 +
                square(square(q[6] - q[4]) + square(q[7] - q[5]) - c2) / c2 +
                square(square(q[6] - q[0]) + square(q[7] - q[1]) - d2) / d2);
    }

    // The CV for the four-bar linkage are the two angles.
    valarray<double> cv() const {
        valarray<double> r(2);

        // If a corner angle t is formed by edges that make angles
        // t1 and t2 w.r.t. the x axis, then sin(t) = sin(t1 - t2) and
        // cos(t) = cos(t1 - t2).
        r[0] = atan2((q[3] - q[1]) * (q[6] - q[0]) - (q[2] - q[0]) * (q[7] - q[1]),
                     (q[2] - q[0]) * (q[6] - q[0]) + (q[3] - q[1]) * (q[7] - q[1]));
        r[1] = atan2((q[5] - q[7]) * (q[6] - q[0]) - (q[4] - q[6]) * (q[7] - q[1]),
                     (q[4] - q[6]) * (q[6] - q[0]) + (q[5] - q[7]) * (q[7] - q[1]));

        return r;
    }
};

int main(int argc, const char* argv[]) {
    // We need to use a ridiculously large number of samples and an even more
    // ridiculous sampling rate if we want a clean free-energy landscape using
    // the Metropolis algorithm.
    auto samples = 500 * 1000;
    auto rate = 15 * 1000;

    FourBar fb;
    assert(fb.energy() == 0);

    // At beta = 10,000, a step size of 0.005 gives ~50% acceptance rate.
    double beta = 10 * 1000.0;
    Metropolis<FourBar> met(fb, beta, "normal", 0.005);

    auto result = met.sample(samples, rate);

    ofstream output;
    if (argc > 1)
        output.open(argv[1]);
    else
        output.open("4bar_met.dat");

    output << scientific;
    for (auto r : result)
        output << r[0] << "\t" << r[1] << endl;

    output.close();
    return 0;
}
