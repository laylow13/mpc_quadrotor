//
// Created by lay on 24-7-17.
//

#include "Euler_mpc.hpp"


Function Euler_mpc::euler_dynamics(model_params_t params_) {
    double g = params_.g;
    double m = params_.m;
    DM J = DM::diag({params_.Jxx, params_.Jyy, params_.Jzz});

    auto f = MX::sym("f", 1);
    auto M = MX::sym("M", 3);
    auto p = MX::sym("p", 3);
    auto v = MX::sym("v", 3);
    auto theta = MX::sym("theta", 1);
    auto phi = MX::sym("phi", 1);
    auto psi = MX::sym("psi", 1);
    auto w = MX::sym("w", 3);

    auto e3 = DM::vertcat({0, 0, 1});
    auto thrust = f * vertcat(cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi),
                              sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi), cos(theta) * cos(phi));
    auto H = MX::vertcat({1, sin(phi) * tan(theta), cos(phi) * tan(theta),
                          0, cos(phi), -sin(phi),
                          0, sin(phi) / cos(theta), cos(phi) / cos(theta)});
    H = reshape(H, 3, 3);

    auto dp = v;
    auto dv = (thrust - m * g * e3) / m;
    auto dq = mtimes(H, w);
    auto dw = mtimes(inv(J), (cross(w, mtimes(J, w)) + M));

    auto state = MX::vertcat({ p, v, theta, phi, psi, w});
    auto input = MX::vertcat({f, M});
    auto state_dot = MX::vertcat({dp, dv, dq, dw});
    return Function("dynamics", {state, input}, {state_dot});
};

Euler_mpc::Euler_mpc(problem_params_t problem_params_, model_params_t model_params_) :
        problem_params(
                problem_params_),
        model_params(
                model_params_) {
    init_mpc_problem();
    init_solver();
};

void Euler_mpc::init_mpc_problem() {
    Function dynamics = euler_dynamics(model_params);
    double ts = problem_params.ts;
    int N = problem_params.N;
    DM &R = problem_params.R;
    DM &Q = problem_params.Q;
    DM &P = problem_params.P;

    MX X0 = MX::sym("X0", 12);  // X0
    MX Xd = MX::sym("Xd", 12, N + 1); // trajectory refence
    MX X = MX::sym("X", 12, N + 1);
    MX U = MX::sym("U", 4, N); // Control vector

    std::vector<double> equal_bound(12, 0);
    std::vector<double> actuator_lb(4, model_params.min_motor_vel * model_params.min_motor_vel);
    std::vector<double> actuator_ub(4, model_params.max_motor_vel * model_params.max_motor_vel);
    DM mixing = compute_mixing(model_params.cf, model_params.ctf, model_params.l);
    if (!lbg.empty()) lbg.clear();
    if (!ubg.empty()) ubg.clear();

    MX cost = 0;    // cost
    std::vector<MX> constraints;
    for (size_t k = 0; k < N; k++) {
        MX Uk = U(Slice(), k);
        MX X_err = X(Slice(), k) - Xd(Slice(), k);
        cost += dot(Uk, mtimes(R, Uk)) + dot(X_err, mtimes(Q, X_err));

        auto Xdot = dynamics({X(Slice(), k), Uk});
        constraints.push_back(X(Slice(), k + 1) - X(Slice(), k) - Xdot[0] * ts);
        lbg = join(lbg, equal_bound);
        ubg = join(ubg, equal_bound);
        constraints.push_back(mtimes(mixing, Uk));
        lbg = join(lbg, actuator_lb);
        ubg = join(ubg, actuator_ub);
    }
    cost += dot(X(Slice(), N) - Xd(Slice(), N), mtimes(P, (X(Slice(), N) - Xd(Slice(), N))));
    constraints.push_back(X(Slice(), 0) - X0);//initial condition constraint
    lbg = join(lbg, equal_bound);
    ubg = join(ubg, equal_bound);

    nlp = {{"x", vertcat(reshape(U, -1, 1), reshape(X, -1, 1))},
           {"p", vertcat(X0, reshape(Xd, -1, 1))},
           {"f", cost},
           {"g", vertcat(constraints)}};
}

DMDict Euler_mpc::compute(const DM &current_state_, const DM &traj_) {
    DMDict arg = {{"p",   vertcat(current_state_, reshape(traj_, -1, 1))},
                  {"lbg", lbg},
                  {"ubg", ubg}};
    return mpc_solver(arg);
}

void Euler_mpc::init_solver(std::string solver_type, std::map<std::string, GenericType> solver_opts) {
    mpc_solver = nlpsol("solver", solver_type, nlp, solver_opts);

}

void Euler_mpc::get_mpc_problem(MXDict &problem_, std::vector<double> &lbg_, std::vector<double> &ubg_) const {
    problem_ = nlp;
    lbg_ = lbg;
    ubg_ = ubg;
}

void
Euler_mpc::set_mpc_problem(const MXDict &problem_, const std::vector<double> &lbg_, const std::vector<double> &ubg_) {
    nlp = problem_;
    lbg = lbg_;
    ubg = ubg_;
}

DM Euler_mpc::compute_mixing(double _cf, double _ctf, double _l) {
    DM effectiveness = DM::zeros(4, 4);
    effectiveness(0, 0) = 1;
    effectiveness(0, 1) = 1;
    effectiveness(0, 2) = 1;
    effectiveness(0, 3) = 1;

    effectiveness(1, 0) = -_l;
    effectiveness(1, 1) = _l;
    effectiveness(1, 2) = _l;
    effectiveness(1, 3) = -_l;

    effectiveness(2, 0) = -_l;
    effectiveness(2, 1) = _l;
    effectiveness(2, 2) = -_l;
    effectiveness(2, 3) = _l;

    effectiveness(3, 0) = -_ctf;
    effectiveness(3, 1) = -_ctf;
    effectiveness(3, 2) = _ctf;
    effectiveness(3, 3) = _ctf;
    effectiveness *= _cf;
    return inv(effectiveness);
}
