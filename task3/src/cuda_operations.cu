#include <iostream>
#include "net_model.h"
#include "cuda_common.h"


//#define DEBUG

__device__ double devCalculateFunctionValue (const double x, const double y)
{
    return (x * x + y * y) * sin(x * y);;
}

__device__ double devCalculateBoundaryValue(const double x, const double y)
{
    return 1 + sin(x * y);
}

void ProcessorsData::InitCudaParameters()
{
    int cudaDeviceNum = 0;
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
    checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaThreadSynchronize());

    checkCudaErrors(cudaSetDevice(cudaDeviceNum));
    checkCudaErrors(cudaGetDeviceProperties(&devProp, 0));



#ifdef DEBUG
    std::cout
         << "multiProcessorCount = " << devProp.multiProcessorCount << endl
         << "maxThreadsPerMultiProcessor = " << devProp.maxThreadsPerMultiProcessor << endl
         << "warpSize = " << devProp.warpSize << endl
         << "maxThreadsPerBlock = " << devProp.maxThreadsPerBlock << endl
         << "maxGridSize = " << devProp.maxGridSize[0] << " " << devProp.maxGridSize[1] << " " << devProp.maxGridSize[2] << endl
         << "maxThreadsDim = " << devProp.maxThreadsDim[0] << " " << devProp.maxThreadsDim[1] << " " << devProp.maxThreadsDim[2] << endl
         << "totalGlobalMem = " << devProp.totalGlobalMem << endl
         << "totalConstMem = " << devProp.totalConstMem << endl
         << "sharedMemPerBlock = " << devProp.sharedMemPerBlock << endl
         << "regsPerBlock = " << devProp.regsPerBlock << endl;
#endif

}


__global__ void kernelAddMatricesAndMultiplyByNumber(double *matrix, double *otherMatrix, double *result,
                                          double alpha, int numRows, int numCols)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows && j < numCols)
    {
        int index = i * numCols + j;
        result[index] = matrix[index] - alpha * otherMatrix[index];
    }
}

void addMatricesAndMultiplyCUDA(ProcessorsData processorData,
                                double* devMatrix, double* devOtherMatrix, double* devResult,
                                double alpha, int numRows, int numCols)
{
    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 blocksPerGrid = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);

    kernelAddMatricesAndMultiplyByNumber<<<blocksPerGrid,threadsPerBlock>>>( devMatrix, devOtherMatrix, devResult,
                                                              alpha, numRows, numCols );
}


