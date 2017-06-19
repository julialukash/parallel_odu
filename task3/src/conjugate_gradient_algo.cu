#include <iostream>

#include "cuda_common.h"
#include "conjugate_gradient_algo.h"
#include "mpi_operations.h"

#define DEBUG

const double eps = 1e-4;

ConjugateGradientAlgo::ConjugateGradientAlgo(const NetModel& modelNet,
                                             const ApproximateOperations& approximateOperations,
                                             const ProcessorsData& processorData):
    processorData(processorData), netModel(modelNet), approximateOperations(approximateOperations)
{
}

ConjugateGradientAlgo::~ConjugateGradientAlgo()
{
    cudaFree( devGroundTruthValues );
}


__global__ void kernelCalculateGroundTruthMatrix(double* devMatrix, const ProcessorsData processorData,
                              const NetModel netModel, double* devXValues, double* devYValues)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int index = i * processorData.ColsCountWithBorders() + j;
    if (i < processorData.RowsCountWithBorders() && j < processorData.ColsCountWithBorders())
    {
        devMatrix[index] =  devCalculateBoundaryValue(netModel.xValue(devXValues, j),
                                                      netModel.yValue(devYValues, i));
    }
}

__global__ void kernelInitBoundaryMatrix(double* devMatrix, const ProcessorsData processorData,
                              const NetModel netModel, double* devXValues, double* devYValues)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int index = i * processorData.ColsCountWithBorders() + j;
    if (i < processorData.RowsCountWithBorders() && j < processorData.ColsCountWithBorders())
    {
        devMatrix[index] = 0.05;
    }
    if (processorData.IsBorderIndices(i, j))
    {
        devMatrix[index] =  devCalculateBoundaryValue(netModel.xValue(devXValues, j),
                                                      netModel.yValue(devYValues, i));
    }
}

__global__ void kernelAbsoluteSubstractCrop(double* devMatrix, double* devOtherMatrix, double* devResMatrix,
                             const ProcessorsData processorData)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= processorData.FirstOwnRowRelativeIndex() && i <= processorData.LastOwnRowRelativeIndex() &&
        j >= processorData.FirstOwnColRelativeIndex() && j <= processorData.LastOwnColRelativeIndex())
    {
        int resIndex = (i - 1) * processorData.ColsCount() + j - 1;
        int index = i * processorData.ColsCountWithBorders() + j;
        devResMatrix[resIndex] = fabs(devMatrix[index] - devOtherMatrix[index]);
    }
}

__global__ void kernelCalculateResiduals(double* devMatrix, double* devResMatrix,
                                    const ProcessorsData processorData,
                                    const NetModel netModel, double* devXValues, double* devYValues)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= processorData.FirstOwnRowRelativeIndex() && i <= processorData.LastOwnRowRelativeIndex() &&
        j >= processorData.FirstOwnColRelativeIndex() && j <= processorData.LastOwnColRelativeIndex())
    {
        int index = i * processorData.ColsCountWithBorders() + j;
        if (processorData.IsBorderIndices(i, j))
        {
            devResMatrix[index] = 0;
        }
        else
        {
            devResMatrix[index] = devMatrix[index] - devCalculateFunctionValue(netModel.xValue(devXValues, j),
                                                                               netModel.yValue(devYValues, i));
        }
    }
}

void ConjugateGradientAlgo::CalculateGroundTruthMatrixCUDA()
{
    checkCudaErrors( cudaMalloc( (void**)&devGroundTruthValues,
                                 processorData.colsCountWithBorders * processorData.rowsCountWithBorders * sizeof(double) ) );

    int numRows = processorData.RowsCountWithBorders();
    int numCols = processorData.ColsCountWithBorders();

    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 blocksPerGrid = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);

    kernelCalculateGroundTruthMatrix<<<blocksPerGrid,threadsPerBlock>>>( devGroundTruthValues, processorData, netModel, devXValues, devYValues);

    return;
}

void ConjugateGradientAlgo::InitCUDA(double* devMatrix)
{
    int numRows = processorData.RowsCountWithBorders();
    int numCols = processorData.ColsCountWithBorders();

    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 blocksPerGrid = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);

    kernelInitBoundaryMatrix<<<blocksPerGrid,threadsPerBlock>>>( devMatrix, processorData, netModel, devXValues, devYValues);

    return;
}


void ConjugateGradientAlgo::RenewBoundsCUDA(double *values)
{
    RenewMatrixBoundRowsCUDA(values, processorData);
    RenewMatrixBoundColsCUDA(values, processorData);
}

bool ConjugateGradientAlgo::IsStopConditionCUDA(double* devP, double* devPreviousP)
{
    int numRows = processorData.RowsCountWithBorders();
    int numCols = processorData.ColsCountWithBorders();
    int numObjectsCropped = (numCols - 2) * (numRows - 2);

    double* devResultMatrix;
    // allocate the memory on the GPU
    checkCudaErrors( cudaMalloc( (void**)&devResultMatrix, numObjectsCropped * sizeof(double) ) );

    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 numBlocks = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);

    kernelAbsoluteSubstractCrop<<<numBlocks,threadsPerBlock>>>( devP, devPreviousP, devResultMatrix, processorData);

    double pDiffNormLocal = approximateOperations.MaxNormValueCUDA(devResultMatrix,
        processorData.rowsCountValue * processorData.colsCountValue, devReduceResultMatrix);
    double pDiffNormGlobal = GetMaxValueFromAllProcessors(pDiffNormLocal);

    cudaFree( devResultMatrix );
    return pDiffNormGlobal < eps;
}

void ConjugateGradientAlgo::CalculateResidualCUDA(double* devP, double* devResiduals)
{
    int numRows = processorData.RowsCountWithBorders();
    int numCols = processorData.ColsCountWithBorders();
    int numObjects = numCols * numRows;

    double* devLaplassP;
    checkCudaErrors( cudaMalloc( (void**)&devLaplassP, numObjects * sizeof(double) ) );

    approximateOperations.CalculateLaplassCUDA(devP, devLaplassP, devXValues, devYValues);

    // allocate the memory on the GPU
    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 blocksPerGrid = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);

    kernelCalculateResiduals<<<blocksPerGrid,threadsPerBlock>>>(devLaplassP, devResiduals, processorData, netModel,
                                                                devXValues, devYValues);

    cudaFree( devLaplassP );

    return;
}

void ConjugateGradientAlgo::CalculateGradientCUDA(double* devResiduals, double* devLaplassResiduals,
                                double* devPreviousGrad, double* devLaplassPreviousGrad,
                                int k, double* devGradient)
{
    if (k == 0)
    {
        checkCudaErrors( cudaMemcpy( devGradient, devResiduals,
                                     processorData.RowsCountWithBorders() * processorData.ColsCountWithBorders() * sizeof(double),
                                     cudaMemcpyDeviceToDevice ) );
    }
    else
    {
        double alpha = CalculateAlphaValueCUDA(devLaplassResiduals, devPreviousGrad, devLaplassPreviousGrad);
        addMatricesAndMultiplyCUDA(processorData, devResiduals, devPreviousGrad, devGradient,
                                   alpha, processorData.RowsCountWithBorders(), processorData.ColsCountWithBorders());
    }
    return;
}

void ConjugateGradientAlgo::CalculateNewPCUDA(double* devPreviousP, double* devGrad, double tau, double* devP)
{
    addMatricesAndMultiplyCUDA(processorData, devPreviousP, devGrad, devP,
                               tau, processorData.RowsCountWithBorders(), processorData.ColsCountWithBorders());
    return;
}

double ConjugateGradientAlgo::CalculateErrorCUDA(double* devP)
{
    int numRows = processorData.RowsCountWithBorders();
    int numCols = processorData.ColsCountWithBorders();
    int numObjectsCropped = (numCols - 2) * (numRows - 2);

    double* devResultMatrix;
    // allocate the memory on the GPU
    checkCudaErrors( cudaMalloc( (void**)&devResultMatrix, numObjectsCropped * sizeof(double) ) );

    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 blocksPerGrid = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);

    kernelAbsoluteSubstractCrop<<<blocksPerGrid, threadsPerBlock>>>( devP, devGroundTruthValues, devResultMatrix, processorData);

    double localError = approximateOperations.MaxNormValueCUDA(devResultMatrix,
        processorData.RowsCount() * processorData.ColsCount(), devReduceResultMatrix);
    double globalError = GetMaxValueFromAllProcessors(localError);

    cudaFree( devResultMatrix );
    return globalError;
}

void ConjugateGradientAlgo::InitNetValues()
{
    // calculate net values
    double* xValues = new double[netModel.xSize];
    double* yValues = new double[netModel.ySize];

    for (int i = netModel.firstColIndex; i <= netModel.lastColIndex + 2; ++i)
    {
        xValues[i - netModel.firstColIndex] = netModel.xMaxBoundary * netModel.f(1.0 * i / (netModel.xPointsCount - 1), 2.0 / 3.0);
    }

    for (int i = netModel.firstRowIndex; i <= netModel.lastRowIndex + 2; ++i)
    {
        yValues[i - netModel.firstRowIndex] = netModel.yMaxBoundary * netModel.f(1.0 * i / (netModel.yPointsCount - 1), 2.0 / 3.0);
    }

    // to gpu
    checkCudaErrors( cudaMemcpy( devXValues, xValues, netModel.xSize * sizeof(double),
                              cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( devYValues, yValues, netModel.ySize * sizeof(double),
                              cudaMemcpyHostToDevice ) );
    delete xValues;
    delete yValues;
}

DoubleMatrix* ConjugateGradientAlgo::ProcessCUDA(double* error)
{
    DoubleMatrix* pMatrix;
    int numRows = processorData.RowsCountWithBorders();
    int numCols = processorData.ColsCountWithBorders();
    int numObjects = numCols * numRows;

    double *devP, *devPreviousP, *devGrad, *devLaplassGrad, *devLaplassPreviousGrad,
           *devResiduals, *devLaplassResiduals, *devPreviousGrad;
    // allocate the memory on the GPU
    checkCudaErrors( cudaMalloc( (void**)&devP, numObjects * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**)&devPreviousP, numObjects * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**)&devGrad, numObjects * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**)&devLaplassGrad, numObjects * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**)&devLaplassPreviousGrad, numObjects * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**)&devResiduals, numObjects * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**)&devLaplassResiduals, numObjects * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**)&devPreviousGrad, numObjects * sizeof(double) ) );

    checkCudaErrors( cudaMalloc( (void**)&devXValues, netModel.xSize  * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**)&devYValues, netModel.ySize  * sizeof(double) ) );

    checkCudaErrors( cudaMalloc( (void**)&devReduceResultMatrix, numObjects * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**)&devAuxiliaryMatrix, numObjects * sizeof(double) ) );


    InitNetValues();

    CalculateGroundTruthMatrixCUDA();

    InitCUDA(devP);

    int iteration = 0;
    double errorValue = -1.0;

    while (true)
    {
        RenewBoundsCUDA(devP);
        // calculate error with ground truth
        errorValue = CalculateErrorCUDA(devP);
        if (processorData.IsMainProcessor())
        {
            std::cout << "it = " << iteration << ", err = " << errorValue << std::endl;
        }

        // check stop condition
        bool stopCondition = iteration != 0 && IsStopConditionCUDA(devP, devPreviousP);

        if (stopCondition)
        {
            DoubleMatrix tmpMatrix = DoubleMatrix(processorData.RowsCountWithBorders(), processorData.ColsCountWithBorders());
            checkCudaErrors( cudaMemcpy( tmpMatrix.matrix, devP,
                                         processorData.RowsCountWithBorders() * processorData.ColsCountWithBorders() * sizeof(double),
                                         cudaMemcpyDeviceToHost ) );

            DoubleMatrix* pCroppedPtr = tmpMatrix.CropMatrix(processorData.FirstOwnRowRelativeIndex(), processorData.RowsCount(),
                                             processorData.FirstOwnColRelativeIndex(), processorData.ColsCount());
            pMatrix = pCroppedPtr;
            break;
        }

        std::swap(devLaplassPreviousGrad, devLaplassGrad);

        CalculateResidualCUDA(devP, devResiduals);
        RenewBoundsCUDA(devResiduals);

        approximateOperations.CalculateLaplassCUDA(devResiduals, devLaplassResiduals, devXValues, devYValues);
        RenewBoundsCUDA(devLaplassResiduals);

        std::swap(devGrad, devPreviousGrad);
        CalculateGradientCUDA(devResiduals, devLaplassResiduals, devPreviousGrad, devLaplassPreviousGrad,
                              iteration, devGrad);

        approximateOperations.CalculateLaplassCUDA(devGrad, devLaplassGrad, devXValues, devYValues);
        RenewBoundsCUDA(devLaplassGrad);

        double tau = CalculateTauValueCUDA(devResiduals, devGrad, devLaplassGrad);
        std::swap(devPreviousP, devP);
        CalculateNewPCUDA(devPreviousP, devGrad, tau, devP);
        ++iteration;
    }
    if (processorData.IsMainProcessor())
    {
//#ifdef DEBUG_MODE
        std::cout << "***** *********** last iteration = " << iteration << ", error = " << errorValue << std::endl;
//#endif
    }

    // memory, you are free now
    checkCudaErrors( cudaFree( devP ) );
    checkCudaErrors( cudaFree( devPreviousP ) );
    checkCudaErrors( cudaFree( devGrad ) );
    checkCudaErrors( cudaFree( devLaplassGrad ) );
    checkCudaErrors( cudaFree( devLaplassPreviousGrad ) );
    checkCudaErrors( cudaFree( devResiduals ) );
    checkCudaErrors( cudaFree( devLaplassResiduals ) );
    checkCudaErrors( cudaFree( devPreviousGrad ) );
    checkCudaErrors( cudaFree( devXValues ) );
    checkCudaErrors( cudaFree( devYValues ) );
    checkCudaErrors( cudaFree( devAuxiliaryMatrix ) );
    checkCudaErrors( cudaFree( devReduceResultMatrix ));

    *error = errorValue;
    return pMatrix;
}

double ConjugateGradientAlgo::CalculateTauValueCUDA(double* devResiduals, double* devGrad, double* devLaplassGrad)
{
    double numerator = approximateOperations.ScalarProductCUDA(devResiduals, devGrad, devXValues, devYValues, devReduceResultMatrix, devAuxiliaryMatrix);
    double denominator = approximateOperations.ScalarProductCUDA(devLaplassGrad, devGrad, devXValues, devYValues, devReduceResultMatrix, devAuxiliaryMatrix);
    double tauValue = GetFractionValueFromAllProcessors(numerator, denominator);
    return tauValue;
}

double ConjugateGradientAlgo::CalculateAlphaValueCUDA(double* devLaplassResiduals, double* devPreviousGrad, double* devLaplassPreviousGrad)
{
    double numerator = approximateOperations.ScalarProductCUDA(devLaplassResiduals, devPreviousGrad, devXValues, devYValues, devReduceResultMatrix, devAuxiliaryMatrix);
    double denominator = approximateOperations.ScalarProductCUDA(devLaplassPreviousGrad, devPreviousGrad, devXValues, devYValues, devReduceResultMatrix, devAuxiliaryMatrix);
    double alphaValue = GetFractionValueFromAllProcessors(numerator, denominator);
    return alphaValue;
}
