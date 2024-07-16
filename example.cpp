#include <casadi/casadi.hpp>

using namespace casadi;

int main() {
//    MX x = MX::sym("x", 2); // Two states
//    MX p = MX::sym("p");    // Free parameter
//
//    // Expression for ODE right-hand side
//    MX z = 1 - pow(x(1), 2);
//    MX rhs = vertcat(z * x(0) - x(1) + 2 * tanh(p), x(0));
//
//    // ODE declaration with free parameter
//    MXDict ode = {{"x", x}, {"p", p}, {"ode", rhs}};
//
//    // Construct a Function that integrates over 1s
//    Function F = integrator("F", "cvodes", ode, 0, 1);
//
//    // Control vector
//    MX u = MX::sym("u", 4, 1);
//
//    x = DM(std::vector<double>{0, 1}); // Initial state
//    for (int k = 0; k < 4; ++k)
//    {
//        // Integrate 1s forward in time:
//        // call integrator symbolically
//        MXDict res = F({{"x0", x}, {"p", u(k)}});
//        x = res["xf"];
//    }
//
//    // NLP declaration
//    MXDict nlp = {{"x", u}, {"f", dot(u, u)}, {"g", x}};
//
//    // Solve using IPOPT
//    Function solver = nlpsol("solver", "ipopt", nlp);
//    DMDict res = solver(DMDict{{"x0", 0.2}, {"lbg", 0}, {"ubg", 0}});
// Create scalar/matrix symbols
    MX x = MX::sym("x", 5, 1);

// Compose into expressions
    MX y = norm_2(x);

// Sensitivity of expression -> new expression
    MX grad_y = gradient(y, x);

// Create a Function to evaluate expression
    Function f = Function("f", {x}, {grad_y});

// Evaluate numerically
    std::vector<DM> grad_y_num = f(DM({1, 2, 3, 4, 5}));
    std::cout << grad_y_num[0];
}
