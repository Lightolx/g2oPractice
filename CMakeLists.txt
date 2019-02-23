cmake_minimum_required(VERSION 3.8)
project(g2o_example)

set(CMAKE_CXX_STANDARD 11)

include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(G2O REQUIRED)

include_directories(${G20_INCLUDE_DIRS})

#add_executable(lineFitting lineFitting.cpp)
#target_link_libraries(lineFitting ${OpenCV_LIBS} g2o_core g2o_stuff)
#
add_executable(poseRetify PoseRectify.cpp)
#SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )
#target_link_libraries(poseRetify ${OpenCV_LIBS} g2o_core g2o_stuff)
target_link_libraries(poseRetify g2o_core g2o_stuff g2o_types_sba g2o_types_slam3d)

add_executable(PoseGraph PoseGraph.cpp)
#SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )
#target_link_libraries(poseRetify ${OpenCV_LIBS} g2o_core g2o_stuff)
target_link_libraries(PoseGraph g2o_core g2o_stuff g2o_types_sba g2o_types_slam3d)