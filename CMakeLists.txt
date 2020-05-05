cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

add_definitions(-std=c++14)
set(CMAKE_CXX_STANDARD 14)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

# Uncomment before debugging/profiling
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

project(camera_fusion)

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES})