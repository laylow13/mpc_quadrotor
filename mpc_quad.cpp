#include <casadi/casadi.hpp>
#include <iostream>

using std::cout;
using namespace casadi;

DM trajectory_gen(double _t, double _ts, size_t _N);

DM compute_mixing(double _cf, double _ctf, double _l);

MX hat(const MX &_v);

MX quat_mult(const MX &_q1, const MX &_q2);

MX quat_rotate_vec(const MX &_q, const MX &_v);

void quat_normalize(MX &_q);

void model_test(Function model) {
    DM test_state = DM(std::vector<double>{0, 0, 0,
                                           0, 0, 0,
                                           1, 0, 0, 0,
                                           0, 0, 0});
    DM test_input = DM({1, 0.01, 0.01, 0.01});
    for (size_t i = 0; i < 10; i++) {
        /* code */
        auto res = model(DMDict{{"x0", test_state},
                                {"p",  test_input}});
        test_state = res["xf"];
        cout << i << ":\n"
             << res["xf"] << "\n";
    }
}

double ts = 0.05;
int N = 60;
// model parameters
double g = 9.8;
double m = 2.;
double cf = 8.54858e-06;
double ctf = 0.016;
double l = 0.174;
double min_motor_vel = 100;
double max_motor_vel = 1000;
DM mixing = compute_mixing(cf, ctf, l);
DM J = DM::diag({0.0217, 0.0217, 0.04});


Function get_quad_model() {// inputs
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
//    Function state_trans = integrator("F", "rk", ode, 0, ts);
//    return state_trans;
}

int main() {

    // simulation settings

    Function quad_model = get_quad_model();
    MX X0 = MX::sym("X0", 13);  // X0
    MX Xd = MX::sym("Xd", 13, N + 1); // trajectory refence
    MX X = MX::sym("X", 13, N + 1);
    MX U = MX::sym("U", 4, N); // Control vector
    DM R = DM::diag({0.0001, 0., 0., 0.});
    DM Q = DM::diag({1., 1., 1.,
                     0., 0., 0.,
                     0.1, 0.1, 0.1, 0.1,
                     0., 0., 0.});
    DM P = DM::diag({1., 1., 1.,
                     0., 0., 0.,
                     0.1, 0.1, 0.1, 0.1,
                     0., 0., 0.});
    MX obj = 0;    // cost
    std::vector<MX> constraints;
    constraints.push_back(X(Slice(), 0) - X0);//initial condition constraint
    std::vector<double> lbg{}, ubg{};
    std::vector<double> equal_bound(13, 0);
    std::vector<double> actuator_lb(4, min_motor_vel * min_motor_vel);
    std::vector<double> actuator_ub(4, max_motor_vel * max_motor_vel);
    lbg = join(lbg, equal_bound);
    ubg = join(ubg, equal_bound);
    for (size_t k = 0; k < N; k++) {
        MX Uk = U(Slice(), k);
        MX X_err = X(Slice(), k) - Xd(Slice(), k);
        obj += dot(Uk, mtimes(R, Uk)) + dot(X_err, mtimes(Q, X_err));
        auto Xdot = quad_model({X(Slice(), k), Uk});
//        auto Xdot = res["xf"];
        constraints.push_back(X(Slice(), k + 1) - X(Slice(), k) - Xdot[0] * ts);
        constraints.push_back(mtimes(mixing, Uk));
        lbg = join(lbg, equal_bound);
        lbg = join(lbg, actuator_lb);
        ubg = join(ubg, equal_bound);
        ubg = join(ubg, actuator_ub);
    }
    obj += dot(X(Slice(), N) - Xd(Slice(), N), mtimes(P, (X(Slice(), N) - Xd(Slice(), N))));

    MXDict nlp = {{"x", vertcat(reshape(U, -1, 1), reshape(X, -1, 1))},
                  {"p", vertcat(X0, reshape(Xd, -1, 1))},
                  {"f", obj},
                  {"g", vertcat(constraints)}}; // TO TEST
    Function quadMPC = nlpsol("solver", "ipopt", nlp, {{"ipopt.print_level", 5}});

    DM initial_state = DM(std::vector<double>{0, 0, 0,
                                              0, 0, 0,
                                              1, 0, 0, 0,
                                              0, 0, 0});
    DM initial_guess = DM(4, N);
    initial_guess(0, Slice()) = m * g;
    initial_guess = reshape(initial_guess, 4 * N, 1);
    DM current_state = initial_state;
    for (size_t i = 0; i < 100; i++) {
        DM traj = trajectory_gen(i * ts, ts, N);
        DMDict arg = {{"p",   vertcat(current_state, reshape(traj, -1, 1))},
                      {"lbg", lbg},
                      {"ubg", ubg}};
        DMDict res = quadMPC(arg);
        MX control_solution = res["x"](Slice(0, 4 * N));
        control_solution = reshape(control_solution, 4, N);
        MX open_traj = res["x"](Slice(4 * N, 4 * N + 13 * (N + 1)));
        open_traj = reshape(open_traj, 13, N + 1);
        cout << "control:" << control_solution << "\n" << "open traj:" << open_traj;
        // break;
//        auto current_input = res["x"](Slice(0, 4));
//        res = state_trans({DMDict{{"x0", current_state}, {"p", current_input}}});
//        current_state = res["xf"];
//        auto err = current_state - traj(Slice(), 0);
//
//        cout << i << ":"
//             << current_input << "\n"
//             << current_state(Slice(0, 3)) << traj(Slice(0, 3), 0) << "\n";
//        cout << "error\n"
//             << err(Slice(0, 3));
//        std::cout << res << std::endl;
        break;
        /* code */
    }
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

DM compute_mixing(double _cf, double _ctf, double _l) {
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

MX hat(const MX &_v) {
    MX _v_hat = MX::zeros(3, 3);
    _v_hat(0, 1) = -_v(2);
    _v_hat(0, 2) = _v(1);
    _v_hat(1, 2) = -_v(0);
    _v_hat(1, 0) = _v(2);
    _v_hat(2, 0) = -_v(1);
    _v_hat(2, 1) = _v(0);
    return _v_hat;
}

MX quat_mult(const MX &_q1, const MX &_q2) {
    if (_q1.size1() != 4 || _q1.size2() != 1 || _q2.size1() != 4 || _q2.size2() != 1)
        throw std::runtime_error("Input quaternions must be 4x1 vectors.");
    MX w1 = _q1(0), x1 = _q1(1), y1 = _q1(2), z1 = _q1(3);
    MX w2 = _q2(0), x2 = _q2(1), y2 = _q2(2), z2 = _q2(3);

    MX w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    MX x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    MX y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    MX z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

    return vertcat(w, x, y, z);
}

MX quat_rotate_vec(const MX &_q, const MX &_v) {
    if (_q.size1() != 4 || _q.size2() != 1 || _v.size1() != 3 || _v.size2() != 1)
        throw std::runtime_error("Input quaternions must be 4x1 vectors.");
    MX _q_conj = vertcat(_q(0), -_q(1), -_q(2), -_q(3));
    MX _v_quat = vertcat(0, _v(0), _v(1), _v(2));
    MX _v_rot = quat_mult(quat_mult(_q, _v_quat), _q_conj);
    return _v_rot(Slice(1, 4));
}

void quat_normalize(MX &_q) {
    if (_q.size1() != 4 || _q.size2() != 1)
        throw std::runtime_error("Input quaternions must be 4x1 vectors.");
    MX _q_norm = norm_1(_q);
    _q = _q / _q_norm;
}