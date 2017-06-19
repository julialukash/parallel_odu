rm -r build
mkdir build
cd build
CC=mpicc CXX=mpicxx cmake ..
make

