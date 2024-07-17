#include <casadi/casadi.hpp>
#include <iostream>
#include "string"
#include "Quaternion_mpc.hpp"

using std::cout;
using namespace casadi;

DM trajectory_gen(double _t, double _ts, size_t _N);

int main() {
    double ts = 0.05;
    int N = 60;
    problem_params_t problem_params{};
    problem_params.ts = ts;
    problem_params.N = N;
    model_params_t model_params{};
    // simulation settings
    Quaternion_mpc controller(problem_params, model_params);

    DM initial_state = DM(std::vector<double>{0, 0, 0,
                                              0, 0, 0,
                                              1, 0, 0, 0,
                                              0, 0, 0});
    DM traj = trajectory_gen(0, ts, N);
    auto res = controller.compute(initial_state, traj);
    MX control_solution = res["x"](Slice(0, 4 * N));
    control_solution = reshape(control_solution, 4, N);
    MX open_traj = res["x"](Slice(4 * N, 4 * N + 13 * (N + 1)));
    open_traj = reshape(open_traj, 13, N + 1);
    cout << "control:" << control_solution << "\n" << "open traj:" << open_traj;
    return 0;
}

DM trajectory_gen(double _t, double _ts, size_t _N) {
    DM traj = DM(13, _N + 1);
    for (size_t i = 0; i <= _N; i++) {
        traj(0, i) = 1;
        traj(1, i) = 1;
        traj(2, i) = 1;
        traj(6, i) = 1;
        traj(7, i) = 0;
        traj(8, i) = 0;
        traj(9, i) = 0;
        // traj(0, i) = cos(_t) - 1;
        // traj(1, i) = sin(_t);
        // traj(2, i) = sin(_t);
        _t += _ts;
    }
    return traj;
}

