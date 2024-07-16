#include <casadi/casadi.hpp>
using namespace casadi;
int main()
{
    MX x = MX::sym("x", 2); // Two states
    MX p = MX::sym("p");    // Free parameter

    // Expression for ODE right-hand side
    MX z = 1 - pow(x(1), 2);
    MX rhs = vertcat(z * x(0) - x(1) + 2 * tanh(p), x(0));

    // ODE declaration with free parameter
    MXDict ode = {{"x", x}, {"p", p}, {"ode", rhs}};

    // Construct a Function that integrates over 1s
    Function F = integrator("F", "cvodes", ode, 0, 1);

    // Control vector
    MX u = MX::sym("u", 4, 1);
    
    x = DM(std::vector<double>{0, 1}); // Initial state
    for (int k = 0; k < 4; ++k)
    {
        // Integrate 1s forward in time:
        // call integrator symbolically
        MXDict res = F({{"x0", x}, {"p", u(k)}});
        x = res["xf"];
    }

    // NLP declaration
    MXDict nlp = {{"x", u}, {"f", dot(u, u)}, {"g", x}};

    // Solve using IPOPT
    Function solver = nlpsol("solver", "ipopt", nlp);
    DMDict res = solver(DMDict{{"x0", 0.2}, {"lbg", 0}, {"ubg", 0}});
}
