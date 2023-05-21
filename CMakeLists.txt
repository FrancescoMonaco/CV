# Set minimun CMake version
cmake_minimum_required(VERSION 2.8)

project(Lab7)

# Set C++ version
set(CMAKE_CXX_STANDARD 11)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

# Add the cpp files to the executable
add_executable(${PROJECT_NAME} Lab7.cpp processing_7.cpp panoramic_utils.cpp)

# Link the OpenCV library to the executable
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
