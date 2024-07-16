#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Define the state vector and dynamics for the quadrotor
struct Quadrotor
{
    Vector3d p;       // Position vector
    Vector3d v;       // Velocity vector
    Matrix3d R;       // Rotation matrix
    Vector3d Omega;   // Angular velocity
    double m;         // Mass (example value)
    Vector3d gravity; // Gravity vector
    Matrix3d J;       // Inertia matrix
    Vector3d T;       // Thrust vector in body frame
    Vector3d M;
    Vector3d Fd; // Drag force
    Vector3d Md; // External disturbance moment

    Quadrotor()
    {
        gravity << 0, 0, -9.81;
        J << 0.0217, 0, 0, 0, 0.0217, 0, 0, 0, 0.04; // Example values
        m = 2.;
    }

    void updateDynamics(double dt)
    {
        // Update position and velocity
        p += dt * v;
        v += dt * (1 / m) * (R * T + m * gravity);

        // Update rotation and angular velocity
        Matrix3d hatOmega;
        hatOmega << 0, -Omega.z(), Omega.y(),
            Omega.z(), 0, -Omega.x(),
            -Omega.y(), Omega.x(), 0;

        R += dt * R * hatOmega;
        Omega += dt * J.inverse() * (Omega.cross(J * Omega) + M);
    }
    void setInput(double f_, const Vector3d &M_)
    {
        T << 0, 0, f_;
        M = M_;
    }
    void printState()
    {
        cout << "p:\n"
             << p << "v:\n"
             << v << "R:\n"
             << Quaterniond(R) << "Omega:\n"
             << Omega << "\n";
    }
};

int main()
{
    // Quadrotor drone;

    // // Example initial conditions
    // drone.p << 0, 0, 0;
    // drone.v << 0, 0, 0;
    // drone.R = Matrix3d::Identity();
    // drone.Omega << 0., 0., 0.;

    // // Simulation parameters
    // double dt = 0.1; // Time step
    // int steps = 100;  // Number of simulation steps

    // for (int i = 0; i < steps; ++i)
    // {
    //     drone.setInput(1, Vector3d{0.01, 0.01, 0.01});
    //     drone.updateDynamics(dt);
    //     if (i % 10 == 0)
    //     {
    //         cout << i << ":\n";
    //         drone.printState();
    //     }
    // }
    Quaterniond q{1, 0.1, 0.2, 0.3};
    Quaterniond w{0, 0.1, 0.1, 0.1};
    auto res = q * w;
    cout << res;
    return 0;
}
