#include <iostream>

#include "cuda_common.h"
#include "approximate_operations.h"
#include "mpi_operations.h"

__global__ void kernelCalculateLaplass(double* devMatrix, double* devResMatrix,
                                    const ProcessorsData processorData,
                                       const NetModel netModel, double* devXValues, double* devYValues)
{
    int numObjects = processorData.colsCountWithBorders * processorData.rowsCountWithBorders;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= processorData.FirstInnerRowRelativeIndex() && i <= processorData.LastInnerRowRelativeIndex() &&
        j >= processorData.FirstInnerColRelativeIndex() && j <= processorData.LastInnerColRelativeIndex())
    {
        int index = i * processorData.ColsCountWithBorders() + j;
        int leftIndex = i * processorData.ColsCountWithBorders() + j - 1;
        int rightIndex = i * processorData.ColsCountWithBorders() + j + 1;
        int upIndex = (i - 1) * processorData.ColsCountWithBorders() + j;
        int downIndex = (i + 1) * processorData.ColsCountWithBorders() + j;
        if (index < numObjects && leftIndex < numObjects && rightIndex < numObjects && downIndex < numObjects && upIndex < numObjects &&
            index > 0 && leftIndex >= 0 && rightIndex >= 0 && upIndex >=0 && downIndex >= 0)
        {
            double xPart = (devMatrix[index] - devMatrix[leftIndex]) / netModel.xStep(devXValues, j - 1) -
                           (devMatrix[rightIndex] - devMatrix[index]) / netModel.xStep(devXValues, j);
            double yPart = (devMatrix[index] - devMatrix[upIndex]) / netModel.yStep(devYValues, i - 1) -
                           (devMatrix[downIndex] - devMatrix[index]) / netModel.yStep(devYValues, i);

            devResMatrix[index] = xPart / netModel.xAverageStep(devXValues, j) +
                                  yPart / netModel.yAverageStep(devYValues, i);
        }
    }
}

__global__ void kernelCalculateMax(double* g_idata, double* g_odata,
                          int n, int numBlocks)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ double data [];


    if (idx < n && blockIdx.x < numBlocks && tid < gridDim.x * blockDim.x)
    {
        double maximum = 0.0;
		// for last values
        for (int k = idx; k < n; k += gridDim.x * blockDim.x)
        {
            maximum = max(maximum, fabs(g_idata[k]));
        }
        data[tid] = maximum;
        __syncthreads ();

        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int nextIndex = (tid + stride) + blockIdx.x * blockDim.x;
            if (((tid % (2 * stride)) == 0) && (tid + stride < blockDim.x) && (nextIndex < n))
            {
                data[tid] = max(data[tid], data[tid + stride]);
            }
            // synchronize within block
            __syncthreads();
        }
        // write result for this block to global mem
        if (tid == 0)
        {
            g_odata[blockIdx.x] = data[0];
        }
    }
}

__global__ void kernelCalculateScalarProductMatrix(double* devMatrix, double* devOtherMatrix, double* devResMatrix,
                                    const ProcessorsData processorData, const NetModel netModel, double* devXValues, double* devYValues)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < processorData.RowsCountWithBorders() && j < processorData.ColsCountWithBorders())
    {
        int index = i * processorData.ColsCountWithBorders() + j;
        devResMatrix[index] = 0;
        if (i >= processorData.FirstInnerRowRelativeIndex() && i <= processorData.LastInnerRowRelativeIndex() &&
            j >= processorData.FirstInnerColRelativeIndex() && j <= processorData.LastInnerColRelativeIndex())
        {
            devResMatrix[index] = devMatrix[index] * devOtherMatrix[index] *
                    netModel.xAverageStep(devXValues, j) * netModel.yAverageStep(devYValues, i);
        }
    }
}

__global__ void kernelCalculateSum(double* g_idata, double* g_odata,
                          int n, int numBlocks)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = tid + blockIdx.x * blockDim.x;

    extern __shared__ double data [];

    if (idx < n && blockIdx.x < numBlocks && tid < gridDim.x * blockDim.x)
    {
        double sum = 0.0;
        // for last values
        for (int k = idx; k < n; k += gridDim.x * blockDim.x)
        {
            sum += g_idata[k];
        }
        data[tid] = sum;
        __syncthreads ();

        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int nextIndex = (tid + stride) + blockIdx.x * blockDim.x;
            if (((tid % (2 * stride)) == 0) && (tid + stride < blockDim.x) && (nextIndex < n))
            {
                data[tid] += data[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0)
        {
            g_odata[blockIdx.x] = data[0];
        }
    }
}

void ApproximateOperations::CalculateLaplassCUDA(double* devCurrentValues, double* devLaplassValues, double* devXValues, double* devYValues) const
{
    int numRows = processorData.RowsCountWithBorders();
    int numCols = processorData.ColsCountWithBorders();

    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 blocksPerGrid = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);

    kernelCalculateLaplass<<<blocksPerGrid,threadsPerBlock>>>(devCurrentValues, devLaplassValues, processorData, netModel,
                                                              devXValues, devYValues);

    return;
}

double ApproximateOperations::MaxNormValueCUDA(double* devMatrix, int numObjects, double *devResultMatrix) const
{

    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks1Dim(numObjects);
    dim3 blocksPerGrid = processorData.GetBlocksPerGrid1Dim(threadsPerBlock, numObjects);
    int numBlocks = blocksPerGrid.x;
    // get max, suggest abs was taken before
    while (true)
    {
        int sharedMemorySize = threadsPerBlock.x * sizeof(double);
        kernelCalculateMax<<<blocksPerGrid,threadsPerBlock, sharedMemorySize>>> ( devMatrix, devResultMatrix, numObjects, numBlocks);

        if (numBlocks == 1)
            break;

        // renew dim3
        threadsPerBlock = processorData.GetThreadsPerBlocks1Dim(numBlocks);
        blocksPerGrid = processorData.GetBlocksPerGrid1Dim(threadsPerBlock, numBlocks);
        numObjects = numBlocks;
        numBlocks = blocksPerGrid.x;
        // swap summing array and result
        std::swap(devMatrix, devResultMatrix);
    }

    double maxValue = 0;
    checkCudaErrors(cudaMemcpy(&maxValue, devResultMatrix, sizeof(double), cudaMemcpyDeviceToHost));

    return maxValue;
}


double ApproximateOperations::ScalarProductCUDA(double *devCurrentValues, double *devOtherValues, double* devXValues, double* devYValues,
                                                double *devResultMatrix, double *devAuxiliaryMatrix) const
{
    int numRows = processorData.RowsCountWithBorders();
    int numCols = processorData.ColsCountWithBorders();
    int numObjects = numCols * numRows;

    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 blocksPerGrid = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);

    kernelCalculateScalarProductMatrix<<<blocksPerGrid,threadsPerBlock>>>(devCurrentValues, devOtherValues, devAuxiliaryMatrix,
                                                                          processorData, netModel, devXValues, devYValues);

//   calculate sum
    threadsPerBlock = processorData.GetThreadsPerBlocks1Dim(numObjects);
    blocksPerGrid = processorData.GetBlocksPerGrid1Dim(threadsPerBlock, numObjects);
    int numBlocks = blocksPerGrid.x;

    while (true)
    {
        int sharedMemorySize = threadsPerBlock.x * sizeof(double);
        kernelCalculateSum<<<blocksPerGrid,threadsPerBlock, sharedMemorySize>>> ( devAuxiliaryMatrix, devResultMatrix, numObjects, numBlocks);

        if (numBlocks == 1)
            break;

        // renew dim3
        threadsPerBlock = processorData.GetThreadsPerBlocks1Dim(numBlocks);
        blocksPerGrid = processorData.GetBlocksPerGrid1Dim(threadsPerBlock, numBlocks);
        numObjects = numBlocks;
        numBlocks = blocksPerGrid.x;

        // swap summing array and result
        std::swap(devAuxiliaryMatrix, devResultMatrix);
    }

    double prodValue = 0;
    checkCudaErrors(cudaMemcpy(&prodValue, devResultMatrix, sizeof(double), cudaMemcpyDeviceToHost));

    return prodValue;
}

ApproximateOperations::ApproximateOperations(const NetModel &model, const ProcessorsData &processorInfo):
    netModel(model), processorData(processorInfo)
{
}
