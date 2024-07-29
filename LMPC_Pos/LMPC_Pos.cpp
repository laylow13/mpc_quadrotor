//
// Created by lay on 24-7-18.
//

#include "LMPC_Pos.hpp"

LMPC_Pos::LMPC_Pos(LMPC_Pos::problem_params_t &problem_params_, LMPC_Pos::model_params_t &model_params_)
        : problem_params(
        problem_params_),
          model_params(
                  model_params_) {
    init_mpc_problem();
    mpc_solver = nlpsol("solver", "ipopt", nlp, {{"ipopt.print_level", 5}});

}

DMDict
LMPC_Pos::compute(const DM &current_state_, const DM &traj_, const DM &U_ref, const DM &U_guess_, const DM &X_guess_) {
    DMDict arg = {{"x0",  vertcat(reshape(U_guess_, -1, 1), reshape(X_guess_, -1, 1))},
                  {"p",   vertcat(current_state_, reshape(traj_, -1, 1), reshape(U_ref, -1, 1))},
                  {"lbg", lbg},
                  {"ubg", ubg}};
    return mpc_solver(arg);
}

void LMPC_Pos::reinit_solver(std::string solver_type, std::map<std::string, GenericType> solver_opts) {
    mpc_solver = nlpsol("solver", solver_type, nlp, solver_opts);
}

void LMPC_Pos::get_mpc_problem(MXDict &problem_, std::vector<double> &lbg_, std::vector<double> &ubg_) const {
    problem_ = nlp;
    lbg_ = lbg;
    ubg_ = ubg;
}

void
LMPC_Pos::set_mpc_problem(const MXDict &problem_, const std::vector<double> &lbg_, const std::vector<double> &ubg_) {
    nlp = problem_;
    lbg = lbg_;
    ubg = ubg_;
}

Function LMPC_Pos::euler_dynamics(LMPC_Pos::model_params_t params_) {
    double g = params_.g;
    double m = params_.m;
    double t_theta = params_.t_theta;
    double t_phi = params_.t_phi;
    double k_theta = params_.k_theta;
    double k_phi = params_.k_phi;

    auto f = MX::sym("f", 1);
    auto theta_d = MX::sym("thetad", 1);
    auto phi_d = MX::sym("phid", 1);
    auto p = MX::sym("p", 3);
    auto v = MX::sym("v", 3);
    auto theta = MX::sym("theta", 1);
    auto phi = MX::sym("phi", 1);
//    auto psi = MX::sym("psi", 1);

    auto e3 = DM::vertcat({0, 0, 1});
    auto thrust = f * vertcat(theta, -phi, 1);

    auto dp = v;
    auto dv = (thrust - m * g * e3) / m;
    auto dtheta = (k_theta * theta_d - theta) / t_theta;
    auto dphi = (k_phi * phi_d - phi) / t_phi;

    auto state = MX::vertcat({p, v, theta, phi});
    auto input = MX::vertcat({f, theta_d, phi_d});
    auto state_dot = MX::vertcat({dp, dv, dtheta, dphi});
    return Function("dynamics", {state, input}, {state_dot});
}

void LMPC_Pos::init_mpc_problem() {
    Function dynamics = euler_dynamics(model_params);
    double ts = problem_params.ts;
    int N = problem_params.N;
    DM &R = problem_params.R;
    DM &Q = problem_params.Q;
    DM &P = problem_params.P;
    double angle_max = problem_params.angle_max;
    double f_max = problem_params.f_max;

    MX X0 = MX::sym("X0", n_state);  // X0
    MX Xd = MX::sym("Xd", n_state, N + 1); // trajectory refence
    MX Ud = MX::sym("Ud", n_input, N);
    MX X = MX::sym("X", n_state, N + 1);
    MX U = MX::sym("U", n_input, N); // Control vector

    std::vector<double> equal_bound(n_state, 0);
    std::vector<double> actuator_lb{0, -angle_max, -angle_max};
    std::vector<double> actuator_ub{f_max, angle_max, angle_max};
    std::vector<double> state_lb{-angle_max, -angle_max};
    std::vector<double> state_ub{angle_max, angle_max};
    if (!lbg.empty()) lbg.clear();
    if (!ubg.empty()) ubg.clear();

    MX cost = 0;    // cost
    std::vector<MX> constraints;
    for (size_t k = 0; k < N; k++) {
        MX Uk = U(Slice(), k);
        MX Xk = X(Slice(), k);
        MX U_err = U(Slice(), k) - Ud(Slice(), k);
        MX X_err = X(Slice(), k) - Xd(Slice(), k);
        cost += dot(U_err, mtimes(R, U_err)) + dot(X_err, mtimes(Q, X_err));
        auto Xdot = dynamics(std::vector<MX>({X(Slice(), k), U(Slice(), k)}));
        constraints.push_back(X(Slice(), k + 1) - X(Slice(), k) - Xdot[0] * ts);
        lbg = join(lbg, equal_bound);
        ubg = join(ubg, equal_bound);
        constraints.push_back(U(Slice(), k));
        lbg = join(lbg, actuator_lb);
        ubg = join(ubg, actuator_ub);
        constraints.push_back(X(Slice(6, 8), k));
        lbg = join(lbg, state_lb);
        ubg = join(ubg, state_ub);
    }
    cost += dot(X(Slice(), N) - Xd(Slice(), N), mtimes(P, (X(Slice(), N) - Xd(Slice(), N))));
    constraints.push_back(X(Slice(), 0) - X0);//initial condition constraint
    lbg = join(lbg, equal_bound);
    ubg = join(ubg, equal_bound);
    constraints.push_back(X(Slice(6, 8), N));
    lbg = join(lbg, state_lb);
    ubg = join(ubg, state_ub);

    nlp = {{"x", vertcat(reshape(U, -1, 1), reshape(X, -1, 1))},
           {"p", vertcat(X0, reshape(Xd, -1, 1), reshape(Ud, -1, 1))},
           {"f", cost},
           {"g", vertcat(constraints)}};
}


