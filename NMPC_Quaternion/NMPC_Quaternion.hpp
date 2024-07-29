//
// Created by lay on 24-7-17.
//
#pragma once

#include <casadi/casadi.hpp>
#include "string"

using namespace casadi;



class NMPC_Quaternion {
public:
    struct problem_params_t {
        double ts = 0.05;
        int N = 60;
        DM R = DM::diag({0.0001, 0., 0., 0.});
        DM Q = DM::diag({1., 1., 1.,
                         0.1, 0.1, 0.1,
                         0.1, 0.1, 0.1, 0.1,
                         0.1, 0.1, 0.1});
        DM P = DM::diag({1., 1., 1.,
                         0.1, 0.1, 0.1,
                         0.1, 0.1, 0.1, 0.1,
                         0.1, 0.1, 0.1});
    };
    struct model_params_t {
        double g = 9.8;
        double m = 2.;
        double cf = 8.54858e-06;
        double ctf = 0.016;
        double l = 0.174;
        double min_motor_vel = 100;
        double max_motor_vel = 1000;
        double Jxx = 0.0217;
        double Jyy = 0.0217;
        double Jzz = 0.04;
    };


    NMPC_Quaternion(problem_params_t problem_params_, model_params_t model_params_);

    DMDict compute(const DM &current_state_, const DM &traj_);

    void init_solver(std::string solver_type = "ipopt",
                     std::map<std::string, GenericType> solver_opts = {{"ipopt.print_level", 5}});

    void get_mpc_problem(MXDict &problem_, std::vector<double> &lbg_, std::vector<double> &ubg_) const;

    void set_mpc_problem(const MXDict &problem_, const std::vector<double> &lbg_, const std::vector<double> &ubg_);

    static Function quaternion_dynamics(model_params_t params_);

    static DM compute_mixing(double _cf, double _ctf, double _l);

    static MX hat(const MX &_v);

    static MX quat_mult(const MX &_q1, const MX &_q2);

    static MX quat_rotate_vec(const MX &_q, const MX &_v);

    static void quat_normalize(MX &_q);



private:
    void init_mpc_problem();


    problem_params_t problem_params;
    model_params_t model_params;
    std::vector<double> lbg{}, ubg{};
    MXDict nlp;
    Function mpc_solver;
};


