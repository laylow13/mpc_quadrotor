cmake_minimum_required(VERSION 3.5.0)
project(mpc_quad VERSION 0.1.0 LANGUAGES C CXX)

find_package(casadi REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(quaternion_mpc STATIC Quaternion_mpc.cpp)
target_link_libraries(quaternion_mpc PUBLIC casadi)

add_executable(example example.cpp)
target_link_libraries(example quaternion_mpc)


