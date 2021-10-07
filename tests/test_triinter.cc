// test_triinter.cc -- program that accepts coordinates of two triangles
//                     as arguments and tells if there is an intersection
//
// This program is meant to be called from the Python testing script.
// This is obviously a very inefficient way of testing the C++ version,
// but I really didn't want to dance with ctypes or write an entirely
// new set of tests for the C++ version.
//
// This file can be compiled with com: https://is.gd/compileanything
//
// com: : ${CXX:=c++}
// com: : ${CXXFLAGS:=-std=c++11 -Wall -Wextra -Werror -fopenmp -Ofast -static}
// com: ${CXX} {} ${CXXFLAGS} -o {.}
//

#include "../utils.hh"
#include <iomanip>

using namespace std;

int main(int argc, const char* argv[]) {
    if (argc < 19) {
        cerr << "usage: " << argv[0] << " A B" << endl;
        return 1;
    }

    vector<valarray<double>> A = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    vector<valarray<double>> B = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    for (auto i = 0; i < 3; i++)
        for (auto j = 0; j < 3; j++)
            A[i][j] = stod(argv[1 + 3*i + j]);

    for (auto i = 0; i < 3; i++)
        for (auto j = 0; j < 3; j++)
            B[i][j] = stod(argv[1 + 3*(i + 3) + j]);

    auto res = triinter(A, B, true);

    cout << res.r << "\t";
    cout << scientific << setprecision(16);
    if (res.r) {
        for (auto x : res.T1)
            cout << x << "\t";
        for (auto x : res.T2)
            cout << x << "\t";
    }

    return 0;
}
