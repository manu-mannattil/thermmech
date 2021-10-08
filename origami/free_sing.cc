// free_sing.cc -- Monte-Carlo integrate the PDF integral near the singularity
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

double U(const double& beta, const double& t, const valarray<double>& x) {
    double u1 = 10 * t * t + 31 * x[0] * x[0] + 18 * x[0] * (2 * t - x[1]) - 12 * t * x[1] +
                3 * x[1] * x[1];
    double u2 = 5 * x[0] * x[0] + 6 * x[0] * (2 * t - x[1]) + (x[1] - 2 * t) * (x[1] - 2 * t);

    return exp(-0.5 * beta * (u1 * u1 / 888 + u2 * u2 / 344));
}

double free(const double& beta, const double& t, const double& a = 10, const size_t& N = 1000) {
    double I = 0;
    for (size_t i = 0; i < N; i++)
        I += U(beta, t, uranarray(2, -a, a));
    I *= 4 * a * a / N;
    return -log(I);
}

int main(int argc, const char* argv[]) {
    size_t N = 1000 * 1000 * 1000;
    double a = 10;
    double beta = 10 * 1000;
    valarray<double> f(75 + 1);

#pragma omp parallel for
    for (auto t = 0; t < 75 + 1; t += 1)
        f[t] = free(beta, t * M_PI / 180, a, N);

    ofstream output;
    if (argc > 1)
        output.open(argv[1]);
    else
        output.open("free_sing_mc.dat");

    for (auto t = 0; t < 75 + 1; t += 1)
        output << t << "\t" << f[t] << endl;

    return 0;
}
