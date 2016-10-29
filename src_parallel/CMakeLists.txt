cmake_minimum_required(VERSION 2.8)

project(task2_m)

set(CMAKE_CXX_STANDARD 11)

set(CXX_STANDARD_REQUIRED)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(SOURCE_LIB ConjugateGradientAlgo.cpp MPIOperations.cpp)

set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive")

set(SOURCE_EXE main.cpp)

add_library(tmp STATIC ${SOURCE_LIB})

add_executable(main ${SOURCE_EXE})

target_link_libraries(main tmp)
