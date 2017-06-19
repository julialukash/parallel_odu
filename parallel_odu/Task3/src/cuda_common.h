#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda.h>
#include <stdio.h>
#include "interface.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


__device__ double devCalculateFunctionValue (const double x, const double y);
__device__ double devCalculateBoundaryValue(const double x, const double y);
__device__ double devCalculateNetXValue(const int i, const int firstColIndex, const int xPointsCount);
__device__ double devCalculateNetYValue(const int i, const int firstRowIndex, const int yPointsCount);
__device__ double devCalculateNetYStep(const int i, const int firstRowIndex, const int yPointsCount);
__device__ double devCalculateNetXStep(const int i, const int firstColIndex, const int xPointsCount);

__device__ double devCalculateNetXAverageStep(const int i, const int firstColIndex, const int xPointsCount);
__device__ double devCalculateNetYAverageStep(const int i, const int firstRowIndex, const int yPointsCount);



void addMatricesAndMultiplyCUDA(ProcessorsData processorData, double *devMatrix, double *devOtherMatrix, double *devResult, double alpha, int numRows, int numCols);



#endif // CUDA_COMMON_H
