// free_quartic.cc -- Monte-Carlo integrate the PDF integral near the singularity
//
// This file can be compiled with com: https://is.gd/compileanything
//
// com: : ${CXX:=c++}
// com: : ${CXXFLAGS:=-std=c++11 -Wall -Wextra -Werror -fopenmp -Ofast -static}
// com: ${CXX} {} ${CXXFLAGS} -o {.}
//

#include "../utils.hh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <valarray>

using namespace std;

// Integral over q4-q7.
double U1(const double& beta, const double& t, const valarray<double>& x) {
    double u1 = 10 * t * t + 31 * x[0] * x[0] + 18 * x[0] * (2 * t - x[1]) - 12 * t * x[1] +
                3 * x[1] * x[1];
    double u2 = -7 * t * t + 389 * x[0] * x[0] + 54 * x[0] * (2 * t - x[1]) - 36 * t * x[1] +
                9 * x[1] * x[1];
    return exp(-0.5 * beta * (u1 * u1 / 888 + u2 * u2 / 649128));
}

// Integral over q7-q10.
double U2(const double& beta, const double& t, const valarray<double>& x) {
    double u1 =
        t * t - 12 * x[0] * x[0] - 20 * t * x[1] - 124 * x[1] * x[1] + 12 * x[0] * (t + 6 * x[1]);
    double u2 = 145 * t * t + 36 * x[0] * x[0] - 1124 * t * x[1] + 1556 * x[1] * x[1] -
                36 * x[0] * (t + 6 * x[1]);
    return exp(-0.5 * beta * (u1 * u1 / 14208 + u2 * u2 / 10386048));
}

valarray<double> free(const double& beta, const double& t, const double& a = 10,
                      const long& N = 1000) {
    // Factor in the front involving DynPerp and other terms.
    double C = beta / (720 * 2.66211);
    double I = 0, I2 = 0;
    double p, err;
    for (auto i = 0; i < N; i++) {
        p = U1(beta, t, uranarray(2, -a, a));
        I += p;
        I2 += p * p;
    }

    I = C * I / N;
    I2 = C * I2 / N;
    err = 4 * a * a * sqrt(I2 - I * I) / sqrt(N);
    I *= 4 * a * a;

    return {t * 180 / M_PI, -log(I), err / I, I, err};
}

int main(int argc, const char* argv[]) {
    long N = 1000 * 1000 * 1000;
    double a = 10;
    double beta = 10 * 1000;
    vector<valarray<double>> res(75 + 1);

#pragma omp parallel for
    for (auto t = 0; t < 75 + 1; t += 1)
        res[t] = free(beta, t * M_PI / 180, a, N);

    ofstream output;
    if (argc > 1)
        output.open(argv[1]);
    else
        output.open("free_quartic.dat");

    for (auto r : res)
        output << r[0] << "\t" << r[1] << "\t" << r[2] << "\t" << r[3] << "\t" << r[4] << endl;

    return 0;
}
