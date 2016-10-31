cmake_minimum_required(VERSION 2.8)

project(task2_lom)

set(CMAKE_CXX_STANDARD 11)

set(CXX_STANDARD_REQUIRED)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

find_package(MPI REQUIRED)
 
include_directories(${MPI_INCLUDE_PATH})
 
SET(CMAKE_CXX_COMPILER mpicc)
 
if(MPI_COMPILE_FLAGS)
  set_target_properties(srcmain PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
 
set(SOURCE_LIB conjugate_gradient_algo.cpp mpi_operations.cpp)

set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive " ${MPI_COMPILE_FLAGS})
 
set(SOURCE_EXE main.cpp)
 
add_library(tmp STATIC ${SOURCE_LIB})
 
add_executable(main ${SOURCE_EXE})
 
target_link_libraries(main tmp)

