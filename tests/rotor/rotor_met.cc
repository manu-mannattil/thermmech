// rotor_met.cc -- stiff rotor in 2D centered around the origin (0, 0).
//
// This file can be compiled and run with com: https://is.gd/compileanything
//
// com: : ${CXX:=c++}
// com: : ${CXXFLAGS:=-std=c++11 -Wall -Wextra -Werror -Ofast -static -I .}
// com: ${CXX} {} ${CXXFLAGS} -o {.}
//

#include "../../utils.hh"
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;
using arr = std::valarray<double>;

class Rotor {
  private:
    double a2;

  public:
    double a;
    arr q;
    Rotor(double a = 1.0) : a(a) {
        a2 = a * a;
        q = {1, 0};
    }

    Rotor(double a, arr q) : Rotor(a) { this->q = q; }

    double energy() const { return pow(((q[0] * q[0] + q[1] * q[1]) - a2), 2); }

    arr cv() const { return {q[0], q[1], atan2(q[1], q[0])}; }
};

int main(int argc, const char* argv[]) {
    // We need to use a ridiculously large number of samples and an even more
    // ridiculous sampling rate if we want a clean free-energy landscape using
    // the Metropolis algorithm.
    auto samples = 500 * 1000;
    auto rate = 100;

    arr q = {0, 1};
    Rotor r(1, q);

    // At beta = 10, a step size of 0.35 gives ~50% acceptance rate.
    double beta = 10;
    Metropolis<Rotor> met(r, beta, "uniform", 0.35, false);
    auto result = met.sample(samples, rate);

    ofstream output;
    if (argc > 1)
        output.open(argv[1]);
    else
        output.open("rotor_met.dat");

    output << scientific;
    for (auto r : result)
        output << r[0] << "\t" << r[1] << "\t" << r[2] << endl;

    output.close();
}
