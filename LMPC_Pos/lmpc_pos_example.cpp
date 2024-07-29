//
// Created by lay on 24-7-18.
//
//
// Created by lay on 24-7-18.
//
#include <casadi/casadi.hpp>
#include <iostream>
#include "string"
#include "LMPC_Pos.hpp"

using std::cout;
using namespace casadi;

DM trajectory_gen(double _t, double _ts, size_t _N);

int main() {
    double ts = 0.02;
    int N = 20;
    LMPC_Pos::problem_params_t problem_params{};
    problem_params.ts = ts;
    problem_params.N = N;

    LMPC_Pos::model_params_t model_params{};
    // simulation settings
    LMPC_Pos controller(problem_params, model_params);
    controller.reinit_solver("ipopt", {{"ipopt.print_level", 0}});

    Function dynamics = LMPC_Pos::euler_dynamics(model_params);
    int n_state = controller.n_state;
    int n_input = controller.n_input;

    DM initial_state = DM(std::vector<double>{0, 0, 0,
                                              0, 0, 0,
                                              0, 0});
    DM current_state = initial_state;
    DM u_ref = DM(n_input, N);
    u_ref(0, Slice()) = model_params.m * model_params.g;
    DM traj = trajectory_gen(0, ts, N);
    DM u_guess = u_ref;
    DM x_guess = DM(n_state, N + 1);
    for (int i = 0; i < 100; ++i) {
        auto res = controller.compute(current_state, traj, u_ref, u_guess, x_guess);
        DM control_solution = res["x"](Slice(0, n_input * N));
        control_solution = reshape(control_solution, n_input, N);
        u_guess = horzcat(control_solution(Slice(), Slice(1, N)), control_solution(Slice(), N - 1));
        DM open_traj = res["x"](Slice(n_input * N, n_input * N + n_state * (N + 1)));
        open_traj = reshape(open_traj, n_state, N + 1);
        x_guess = horzcat(open_traj(Slice(), Slice(1, N + 1)), open_traj(Slice(), N));
        auto state_dot = dynamics(std::vector<DM>({current_state, control_solution(Slice(), 0)}));
        current_state += state_dot[0] * ts;
    }
    cout << current_state;
    return 0;
}

DM trajectory_gen(double _t, double _ts, size_t _N) {
    DM traj = DM(8, _N + 1);
    for (size_t i = 0; i <= _N; i++) {
        traj(0, i) = 1;
        traj(1, i) = 1;
        traj(2, i) = 1;
        traj(3, i) = 0;
        traj(4, i) = 0;
        traj(5, i) = 0;
        traj(6, i) = 0;
        traj(7, i) = 0;
        _t += _ts;
    }
    return traj;
}

