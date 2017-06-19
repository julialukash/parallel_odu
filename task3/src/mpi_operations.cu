#include <iostream>

#include "mpi_operations.h"
#include "cuda_common.h"

__global__ void kernelGetMatrixRow(double* matrix, double *result,
                                int startRowIndex, int numRows, int numCols)
{

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows && j < numCols && i == startRowIndex)
    {
        int relativeIndex = startRowIndex * numCols + j;
        int index = j;
        result[index] = matrix[relativeIndex];
    }
}

__global__ void kernelSetMatrixRow(double* matrix, double *result,
                                int startRowIndex, int numRows, int numCols)
{

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows && j < numCols && i == startRowIndex)
    {
        int relativeIndex = startRowIndex * numCols + j;
        int index = j;
        matrix[relativeIndex] = result[index];
    }
}


void RenewMatrixBoundRowsCUDA(double* devValues, const ProcessorsData& processorData)
{
    MPI_Status status;
    int downProcessorRank = processorData.IsLastProcessor() ? MPI_PROC_NULL : processorData.down;
    int upProcessorRank = processorData.IsFirstProcessor() ? MPI_PROC_NULL : processorData.up;

    int numCols = processorData.colsCountWithBorders;
    int numRows = processorData.rowsCountWithBorders;

    // get first and last row
    double* borderRows = new double[numCols * 2];
    double* ownRows = new double[numCols * 2];


    double *devRow;
    checkCudaErrors( cudaMalloc( (void**)&devRow, numCols * sizeof(double) ) );

    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 numBlocks = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);

    // copy own rows to host
    kernelGetMatrixRow<<<numBlocks,threadsPerBlock>>>( devValues, devRow,
                                                       processorData.FirstOwnRowRelativeIndex(), numRows, numCols );
    checkCudaErrors( cudaMemcpy( &(ownRows[0]), devRow, numCols * sizeof(double), cudaMemcpyDeviceToHost ) );

    kernelGetMatrixRow<<<numBlocks,threadsPerBlock>>>( devValues, devRow,
                                                    processorData.LastOwnRowRelativeIndex(), numRows, numCols );
    checkCudaErrors( cudaMemcpy( &(ownRows[numCols]), devRow, numCols * sizeof(double), cudaMemcpyDeviceToHost ) );


    // send to next processor last "no border" line
    // receive from prev processor first "border" line
    MPI_Sendrecv(&(ownRows[processorData.colsCountWithBorders]), processorData.ColsCountWithBorders(), MPI_DOUBLE, downProcessorRank, UP,
                 &(borderRows[0]), processorData.ColsCountWithBorders(), MPI_DOUBLE, upProcessorRank, UP,
                 processorData.gridComm, &status);
    // send to prev processor first "no border" line
    // receive from next processor last "border" line
    MPI_Sendrecv(&(ownRows[0]), processorData.ColsCountWithBorders(), MPI_DOUBLE, upProcessorRank, DOWN,
                 &(borderRows[processorData.ColsCountWithBorders()]), processorData.ColsCountWithBorders(), MPI_DOUBLE, downProcessorRank, DOWN,
                 processorData.gridComm, &status);


    // copy renewed border rows to device
    checkCudaErrors( cudaMemcpy( devRow, &(borderRows[0]), numCols * sizeof(double),
                              cudaMemcpyHostToDevice ) );
    kernelSetMatrixRow<<<numBlocks,threadsPerBlock>>>( devValues, devRow,
                                                    0, numRows, numCols );

    checkCudaErrors( cudaMemcpy( devRow, &(borderRows[numCols]), numCols * sizeof(double),
                              cudaMemcpyHostToDevice ) );
    kernelSetMatrixRow<<<numBlocks,threadsPerBlock>>>( devValues, devRow,
                                                    numRows - 1, numRows, numCols );

    cudaFree( devRow );
    delete borderRows;
    delete ownRows;
}


__global__ void kernelGetMatrixColumn(double* matrix, double *result,
                                int startColIndex, int numRows, int numCols)
{

    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < numRows && j < numCols && j == startColIndex)
    {
        int relativeIndex = i * numCols + startColIndex;
        int index = i;
        result[index] = matrix[relativeIndex];
    }
}


__global__ void kernelSetMatrixColumn(double* matrix, double *result,
                                int startColIndex, int numRows, int numCols)
{

    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < numRows && j < numCols && j == startColIndex)
    {
        int relativeIndex = i * numCols + startColIndex;
        int index = i;
        matrix[relativeIndex] = result[index];
    }
}


void RenewMatrixBoundColsCUDA(double* devValues, const ProcessorsData& processorData)
{
    MPI_Status status;
    int leftProcessorRank = processorData.IsLeftProcessor() ? MPI_PROC_NULL : processorData.left;
    int rightProcessorRank = processorData.IsRightProcessor() ? MPI_PROC_NULL : processorData.right;

    int numCols = processorData.colsCountWithBorders;
    int numRows = processorData.rowsCountWithBorders;

    // get first and last row
    double* borderCols = new double[numRows * 2];
    double* ownCols = new double[numRows * 2];

    double* devColumn;

    checkCudaErrors( cudaMalloc( (void**)&devColumn, numRows * sizeof(double) ) );

    dim3 threadsPerBlock = processorData.GetThreadsPerBlocks2Dim(numCols, numRows);
    dim3 numBlocks = processorData.GetBlocksPerGrid2Dim(threadsPerBlock, numCols, numRows);


    kernelGetMatrixColumn<<<numBlocks,threadsPerBlock>>>( devValues, devColumn,
                                                       processorData.FirstOwnColRelativeIndex(), numRows, numCols );
    checkCudaErrors( cudaMemcpy( &(ownCols[0]), devColumn, numRows * sizeof(double), cudaMemcpyDeviceToHost ) );

    kernelGetMatrixColumn<<<numBlocks,threadsPerBlock>>>( devValues, devColumn,
                                                       processorData.LastOwnColRelativeIndex(), numRows, numCols );
    checkCudaErrors( cudaMemcpy( &(ownCols[numRows]), devColumn, numRows * sizeof(double), cudaMemcpyDeviceToHost ) );


    // send to right processor last "no border" line
    // receive from left processor first "border" line
    MPI_Sendrecv(&(ownCols[numRows]), numRows, MPI_DOUBLE, rightProcessorRank, LEFT,
                 &(borderCols[0]), numRows, MPI_DOUBLE, leftProcessorRank, LEFT,
                 processorData.gridComm, &status);

    // send to left processor first "no border" line
    // receive from next processor last "border" line
    MPI_Sendrecv(&(ownCols[0]), numRows, MPI_DOUBLE, leftProcessorRank, RIGHT,
                 &(borderCols[numRows]), numRows, MPI_DOUBLE, rightProcessorRank, RIGHT,
                 processorData.gridComm, &status);



    checkCudaErrors( cudaMemcpy( devColumn, &(borderCols[0]), numRows * sizeof(double),
                              cudaMemcpyHostToDevice ) );
    kernelSetMatrixColumn<<<numBlocks,threadsPerBlock>>>( devValues, devColumn,
                                                    0, numRows, numCols );
    checkCudaErrors( cudaMemcpy( devColumn, &(borderCols[numRows]), numRows * sizeof(double),
                              cudaMemcpyHostToDevice ) );
    kernelSetMatrixColumn<<<numBlocks,threadsPerBlock>>>( devValues, devColumn,
                                                    numCols - 1, numRows, numCols );

    cudaFree( devColumn );
    delete borderCols;
    delete ownCols;
}

const int ndims = 2;

int SplitFunction(int N0, int N1, int p)
// This is the splitting procedure of proc. number p. The integer p0
// is calculated such that abs(N0/p0 - N1/(p-p0)) --> min.
{
    float n0, n1;
    int p0, i;

    n0 = (float) N0; n1 = (float) N1;
    p0 = 0;

    for(i = 0; i < p; i++)
    {
        if(n0 > n1)
        {
            n0 = n0 / 2.0;
            ++p0;
        }
        else
        {
            n1 = n1 / 2.0;
        }
    }
    return p0;
}

ProcessorsData* CreateProcessorData(int processorsCount, int N0, int N1, int power)
{
    MPI_Comm gridComm;             // this is a handler of a new communicator.
    int Coords[2];
    int periods[2] = {0,0};         // it is used for creating processes topology.
    int rank, left, right, up, down;

    int p0 = SplitFunction(N0, N1, power);
    int p1 = power - p0;

    ProcessorsData* processorInfoPtr = new ProcessorsData(processorsCount);
    processorInfoPtr->InitCartParameters(p0, p1, N0, N1);

    // the cartesian topology of processes is being created ...
    MPI_Cart_create(MPI_COMM_WORLD, ndims, processorInfoPtr->dims, periods, true, &gridComm);
    MPI_Comm_rank(gridComm, &rank);
    MPI_Cart_coords(gridComm, rank, ndims, Coords);

    MPI_Cart_shift(gridComm, 0, 1, &left, &right);
    MPI_Cart_shift(gridComm, 1, 1, &down, &up);

    processorInfoPtr->left = left;       processorInfoPtr->up = up;
    processorInfoPtr->right = right;     processorInfoPtr->down = down;

    // init processors with their part of data
    processorInfoPtr->rank = rank;
    processorInfoPtr->gridComm = gridComm;
    processorInfoPtr->iCartIndex = Coords[0];
    processorInfoPtr->jCartIndex = Coords[1];
    processorInfoPtr->InitProcessorRowsParameters();
    processorInfoPtr->InitProcessorColsParameters();

    processorInfoPtr->InitCudaParameters();
    return processorInfoPtr;
}

double GetMaxValueFromAllProcessors(double localValue)
{
    double globalValue;
    MPI_Allreduce(&localValue, &globalValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return globalValue;
}

double GetFractionValueFromAllProcessors(double numerator, double denominator)
{
    double localValue[2] = {numerator, denominator};
    double globalValue[2] = {0, 0};
    MPI_Allreduce(localValue, globalValue, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return globalValue[0] / globalValue[1];
}

