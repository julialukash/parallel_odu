cd build
rm slurm*
rm main
nvcc -rdc=true -arch=sm_20 -ccbin mpicxx ../main.cpp ../cuda_operations.cu ../conjugate_gradient_algo.cu ../approximate_operations.cu ../mpi_operations.cu -o main -Xcompiler -std=c++98

