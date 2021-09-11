// origami_met.cc -- sample the origami shape space using Metropolis MC.
//
// This file can be compiled with com: https://is.gd/compileanything
//
// com: : ${CXX:=c++}
// com: : ${CXXFLAGS:=-std=c++11 -Wall -Wextra -Werror -Ofast -static -I .}
// com: ${CXX} {} ${CXXFLAGS} -o {.}
//

#include "../utils.hh"
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

// Origami made out of a unit square with one corner at the origin, sides along
// the axes, and internal vertices (a, b) and (c, d).
class Origami {
  private:
    valarray<double> l2; // length-square of all bars
    valarray<double> tt; // fold angles

    // Do the doubles a and b have different signs?
    bool diffsign(const double& a, const double& b) {
        if ((a >= 0 && b >= 0) || (a < 0 && b < 0))
            return false;
        else
            return true;
    }

    // Reshape q into a valarray of valarrays, each belonging to R^3.  There
    // are fancier ways of reshaping a valarray, but since ours isn't very big,
    // brute forcing works just fine.
    valarray<valarray<double>> joints() const {
        valarray<valarray<double>> r = {{q[0], q[1], q[2]},    {q[3], q[4], q[5]},
                                        {q[6], q[7], q[8]},    {q[9], q[10], q[11]},
                                        {q[12], q[13], q[14]}, {q[15], q[16], q[17]}};
        return r;
    }

    // Length-square map: R^18 -> R^11.
    valarray<double> len2(const valarray<valarray<double>>& r) const {
        valarray<double> l(11);
        l[0] = dot(r[0] - r[1], r[0] - r[1]);
        l[1] = dot(r[2] - r[1], r[2] - r[1]);
        l[2] = dot(r[3] - r[2], r[3] - r[2]);
        l[3] = dot(r[3] - r[0], r[3] - r[0]);
        l[4] = dot(r[4] - r[0], r[4] - r[0]);
        l[5] = dot(r[4] - r[2], r[4] - r[2]);
        l[6] = dot(r[4] - r[3], r[4] - r[3]);
        l[7] = dot(r[5] - r[0], r[5] - r[0]);
        l[8] = dot(r[5] - r[1], r[5] - r[1]);
        l[9] = dot(r[5] - r[2], r[5] - r[2]);
        l[10] = dot(r[5] - r[4], r[5] - r[4]);
        return l;
    }

  public:
    double a, b, c, d;
    valarray<double> q;
    Origami(const double& a = 0.25, const double& b = 0.50, const double& c = 0.75,
            const double& d = 0.50)
        : a(a), b(b), c(c), d(d) {
        q = {0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., a, b, 0., c, d, 0.};
        l2 = len2(joints());
        tt = angles(joints());
    }

    Origami(const double& a, const double& b, const double& c, const double& d,
            const valarray<double>& q)
        : Origami{a, b, c, d} {
        this->q = q;
    }

    // Compute the fold angles of all 7 folds when given the joints.
    valarray<double> angles(const valarray<valarray<double>>& r) const {
        valarray<double> aa(7);

        // Folds are ordered counterclockwise around internal vertices 5 and 6.
        aa[0] = normangle(r[2] - r[4], r[5] - r[4], r[3] - r[4]); // fold 3 - 5
        aa[1] = normangle(r[3] - r[4], r[2] - r[4], r[0] - r[4]); // fold 4 - 5
        aa[2] = normangle(r[0] - r[4], r[3] - r[4], r[5] - r[4]); // fold 1 - 5
        aa[3] = normangle(r[5] - r[4], r[0] - r[4], r[2] - r[4]); // fold 6 - 5
        aa[4] = normangle(r[0] - r[5], r[4] - r[5], r[1] - r[5]); // fold 1 - 6
        aa[5] = normangle(r[1] - r[5], r[0] - r[5], r[2] - r[5]); // fold 2 - 6
        aa[6] = normangle(r[2] - r[5], r[1] - r[5], r[4] - r[5]); // fold 3 - 6

        return aa;
    }

    // CV is just the angles.
    valarray<double> cv() const { return tt; }

    // Energy of the origami.
    double energy() {
        valarray<valarray<double>> r = joints();
        valarray<double> aa = angles(r);

        // If at least one angle has changed sign and if the new/old angle is
        // larger than 100, return NaN because it indicates a face crossing.
        for (auto i = 0; i < 7; i++)
            if (diffsign(aa[i], tt[i]) && (abs(aa[i]) > 100 || abs(tt[i]) > 100))
                return nan("");

        // Update the angles before returning.  New angles would differ from
        // old angles only if q has changed.
        tt = aa;
        auto e = 0.125 * ((len2(r) - l2) * (len2(r) - l2) / l2).sum();
        return e;
    }
};

int main(int argc, const char* argv[]) {
    // We need to use a ridiculously large number of samples and an even more
    // ridiculous sampling rate if we want a clean free-energy landscape using
    // the Metropolis algorithm.
    auto samples = 500 * 1000;
    auto rate = 15 * 1000;

    Origami og;
    assert(og.energy() == 0);

    // At beta = 10,000, a step size of 0.003 gives ~50% acceptance rate.
    auto beta = 10 * 1000.0;
    Metropolis<Origami> met(og, beta, "normal", 0.003);

    auto result = met.sample(samples, rate);

    ofstream output;
    if (argc > 1)
        output.open(argv[1]);
    else
        output.open("origami_met.dat");

    output << scientific;
    for (auto r : result) {
        for (auto i = 0; i < 7; i++) {
            output << r[i];
            if (i < 6)
                output << "\t";
        }
        output << endl;
    }

    output.close();
    return 0;
}
