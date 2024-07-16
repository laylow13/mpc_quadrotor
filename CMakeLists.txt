cmake_minimum_required(VERSION 3.5.0)
project(mpc_quad VERSION 0.1.0 LANGUAGES C CXX)

find_package(casadi REQUIRED)
find_package(Eigen3 REQUIRED)
add_executable(mpc_quad mpc_quad.cpp)
target_link_libraries(mpc_quad casadi)

add_executable(example example.cpp)
target_link_libraries(example casadi)

add_executable(quad_dynamic_sim quad_dynamics.cpp)
target_link_libraries(quad_dynamic_sim Eigen3::Eigen)