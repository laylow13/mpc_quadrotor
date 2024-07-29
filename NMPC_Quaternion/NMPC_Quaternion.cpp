//
// Created by lay on 24-7-17.
//

#include "NMPC_Quaternion.hpp"

NMPC_Quaternion::NMPC_Quaternion(problem_params_t problem_params_, model_params_t model_params_) :
        problem_params(
                problem_params_),
        model_params(
                model_params_) {
    init_mpc_problem();
    init_solver();
};


void NMPC_Quaternion::init_mpc_problem() {
    Function dynamics = quaternion_dynamics(model_params);
    double ts = problem_params.ts;
    int N = problem_params.N;
    DM &R = problem_params.R;
    DM &Q = problem_params.Q;
    DM &P = problem_params.P;

    MX X0 = MX::sym("X0", 13);  // X0
    MX Xd = MX::sym("Xd", 13, N + 1); // trajectory refence
    MX X = MX::sym("X", 13, N + 1);
    MX U = MX::sym("U", 4, N); // Control vector

    std::vector<double> equal_bound(13, 0);
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

void NMPC_Quaternion::init_solver(std::string solver_type, std::map<std::string, GenericType> solver_opts) {
    mpc_solver = nlpsol("solver", solver_type, nlp, solver_opts);
}

void NMPC_Quaternion::get_mpc_problem(MXDict &problem_, std::vector<double> &lbg_, std::vector<double> &ubg_) const {
    problem_ = nlp;
    lbg_ = lbg;
    ubg_ = ubg;
}

void
NMPC_Quaternion::set_mpc_problem(const MXDict &problem_, const std::vector<double> &lbg_,
                                 const std::vector<double> &ubg_) {
    nlp = problem_;
    lbg = lbg_;
    ubg = ubg_;
}

Function NMPC_Quaternion::quaternion_dynamics(model_params_t params_) {// inputs
    double g = params_.g;
    double m = params_.m;
    DM J = DM::diag({params_.Jxx, params_.Jyy, params_.Jzz});

    MX f = MX::sym("f");
    MX M = MX::sym("M", 3);
    // state variables
    MX p = MX::sym("p", 3);
    MX v = MX::sym("v", 3);
    MX q = MX::sym("q", 4);
    MX w = MX::sym("w", 3);

    // intemediate variables
    DM e3 = DM::vertcat({0, 0, 1});
    MX thrust = quat_rotate_vec(q, f * e3);
    // dynamics
    MX dp = v;
    MX dv = (thrust - m * g * e3) / m;
    MX dq = 0.5 * quat_mult(q, vertcat(0, w));
    MX dw = mtimes(inv(J), (cross(w, mtimes(J, w)) + M));
    MX rhs = vertcat(dp, dv, dq, dw);
    MX state = vertcat(p, v, q, w);
    MX input = vertcat(f, M);
    MXDict ode = {{"x",   state},
                  {"p",   input},
                  {"ode", rhs}};
    return Function("dynamics", {state, input}, {rhs});
}


DMDict NMPC_Quaternion::compute(const DM &current_state_, const DM &traj_) {
    DMDict arg = {{"p",   vertcat(current_state_, reshape(traj_, -1, 1))},
                  {"lbg", lbg},
                  {"ubg", ubg}};
    return mpc_solver(arg);
}

DM NMPC_Quaternion::compute_mixing(double _cf, double _ctf, double _l) {
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

MX NMPC_Quaternion::hat(const MX &_v) {
    MX _v_hat = MX::zeros(3, 3);
    _v_hat(0, 1) = -_v(2);
    _v_hat(0, 2) = _v(1);
    _v_hat(1, 2) = -_v(0);
    _v_hat(1, 0) = _v(2);
    _v_hat(2, 0) = -_v(1);
    _v_hat(2, 1) = _v(0);
    return _v_hat;
}

MX NMPC_Quaternion::quat_mult(const MX &_q1, const MX &_q2) {
    if (_q1.size1() != 4 || _q1.size2() != 1 || _q2.size1() != 4 || _q2.size2() != 1)
        throw std::runtime_error("InputVar quaternions must be 4x1 vectors.");
    MX w1 = _q1(0), x1 = _q1(1), y1 = _q1(2), z1 = _q1(3);
    MX w2 = _q2(0), x2 = _q2(1), y2 = _q2(2), z2 = _q2(3);

    MX w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    MX x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    MX y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    MX z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

    return vertcat(w, x, y, z);
}

MX NMPC_Quaternion::quat_rotate_vec(const MX &_q, const MX &_v) {
    if (_q.size1() != 4 || _q.size2() != 1 || _v.size1() != 3 || _v.size2() != 1)
        throw std::runtime_error("InputVar quaternions must be 4x1 vectors.");
    MX _q_conj = vertcat(_q(0), -_q(1), -_q(2), -_q(3));
    MX _v_quat = vertcat(0, _v(0), _v(1), _v(2));
    MX _v_rot = quat_mult(quat_mult(_q, _v_quat), _q_conj);
    return _v_rot(Slice(1, 4));
}

void NMPC_Quaternion::quat_normalize(MX &_q) {
    if (_q.size1() != 4 || _q.size2() != 1)
        throw std::runtime_error("InputVar quaternions must be 4x1 vectors.");
    MX _q_norm = norm_1(_q);
    _q = _q / _q_norm;
}
