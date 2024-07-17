//
// Created by lay on 24-7-17.
//
#include <nlopt.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Eigen>
#include "vector"

using namespace std;
using namespace autodiff;
using namespace Eigen;
using StateVar = Vector<var, 13>;
using InputVar = Vector<var, 4>;
struct state {
    Vector3var p;
    Vector3var v;
    Quaternion<var> q;
    Vector3var w;
};
struct Params {
    Matrix<double, 13, 13> Q;
    Matrix<double, 13, 13> P;
    Matrix<double, 4, 4> R;
};

var f(const vector<StateVar> &x, const vector<StateVar> &xd, const vector<InputVar> &u, const Params &params) {
    const Matrix<double, 13, 13> &Q = params.Q;
    const Matrix<double, 13, 13> &P = params.P;
    const Matrix<double, 4, 4> &R = params.R;
    if (not(x.size() == xd.size() and x.size() == (u.size() + 1))) {
        throw;
    }
    int N = u.size();
    StateVar x_err = x[N] - xd[N];
    var cost = x_err.dot(P * x_err);
    for (int i = 0; i < N; ++i) {
        x_err = x[i] - xd[i];
        cost += x_err.dot(Q * x_err) + u[i].dot(R * u[i]);
    }
}

var dynamics() {

}

double cost(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
    auto state_first = x.begin();
    int N = 10;
    vector<StateVar> x_var_list;
    for (int i = 0; i < N + 1; ++i) {
        const vector<double> x_1(state_first + i * N, state_first + (i + 1) * N);
        StateVar x_var = Eigen::Map<StateVar, Eigen::Unaligned>((var *) x_1.data(), x_1.size());
        x_var_list.push_back(x_var);
    }

}

int main() {

    return 0;
}