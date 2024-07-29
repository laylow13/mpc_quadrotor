//
// Created by lay on 24-7-18.
//
#pragma once

#include "casadi/casadi.hpp"

using namespace casadi;

class LMPC_Pos {
public:
    struct problem_params_t {
        double ts = 0.05;
        int N = 60;
        DM R = DM::diag({0.01, 0.01, 0.01});
        DM Q = DM::diag({1., 1., 1.,
                         0.1, 0.1, 0.1,
                         0.1, 0.1});
        DM P = DM::diag({1., 1., 1.,
                         0.1, 0.1, 0.1,
                         0.1, 0.1});
        double angle_max = pi / 6;//30 degree
        double f_max = 30;
    };
    struct model_params_t {
        double g = 9.8;
        double m = 2.;
        double t_theta = 0.1;
        double t_phi = 0.1;
        double k_theta = 1.;
        double k_phi = 1.;
    };

    LMPC_Pos(LMPC_Pos::problem_params_t &problem_params_, LMPC_Pos::model_params_t &model_params_);

    DMDict compute(const DM &current_state_, const DM &traj_, const DM &U_ref, const DM &U_guess_, const DM &X_guess_);

    void reinit_solver(std::string solver_type = "ipopt",
                       std::map<std::string, GenericType> solver_opts = {{"ipopt.print_level", 5}});

    void get_mpc_problem(MXDict &problem_, std::vector<double> &lbg_, std::vector<double> &ubg_) const;

    void set_mpc_problem(const MXDict &problem_, const std::vector<double> &lbg_, const std::vector<double> &ubg_);

    static Function euler_dynamics(model_params_t params_);

    const int n_state = 8;
    const int n_input = 3;

private:
    void init_mpc_problem();


    problem_params_t problem_params;
    model_params_t model_params;
    std::vector<double> lbg{}, ubg{};
    MXDict nlp;
    Function mpc_solver;
};

