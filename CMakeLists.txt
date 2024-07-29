cmake_minimum_required(VERSION 3.5.0)
project(mpc_quad VERSION 0.1.0 LANGUAGES C CXX)

find_package(casadi REQUIRED)

add_library(nmpc_quaternion STATIC NMPC_Quaternion/NMPC_Quaternion.cpp)
target_link_libraries(nmpc_quaternion PUBLIC casadi)

add_executable(quaternion_example NMPC_Quaternion/quaternion_example.cpp)
target_link_libraries(quaternion_example nmpc_quaternion)

add_library(nmpc_euler STATIC NMPC_Euler/NMPC_Euler.cpp )
target_link_libraries(nmpc_euler PUBLIC casadi)

add_executable(euler_example NMPC_Euler/euler_example.cpp)
target_link_libraries(euler_example nmpc_euler)

add_library(lmpc_full LMPC_Full/LMPC_Full.cpp LMPC_Full/LMPC_Full.hpp)
target_link_libraries(lmpc_full PUBLIC casadi)

add_executable(lmpc_full_example LMPC_Full/lmpc_full_example.cpp)
target_link_libraries(lmpc_full_example PUBLIC lmpc_full)

add_library(lmpc_pos  LMPC_Pos/LMPC_Pos.cpp  LMPC_Pos/LMPC_Pos.hpp)
target_link_libraries(lmpc_pos PUBLIC casadi)

add_executable(lmpc_pos_example LMPC_Pos/lmpc_pos_example.cpp)
target_link_libraries(lmpc_pos_example lmpc_pos)