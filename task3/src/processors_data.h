#ifndef PROCESSORSDATA_H
#define PROCESSORSDATA_H

#include "interface.h"
#include <mpi.h>
#include "cuda_runtime.h"


using std::endl;

class ProcessorsData
{
private:
public:
    struct cudaDeviceProp  devProp;
    int rank, processorsCount;
    int iCartIndex, jCartIndex;
    int startRowIndex, endRowIndex, rowsCountValue, rowsCountWithBorders;
    int startColIndex, endColIndex, colsCountValue, colsCountWithBorders;
    int left, right, down, up;
    int n1, k1, n0, k0, N0, N1;
    int dims[2];
    MPI_Comm gridComm;


    ProcessorsData(int processorsCountValue): processorsCount(processorsCountValue){ }

    // almost only here
    __host__ __device__ inline bool IsMainProcessor() const { return rank == 0; }
    __host__ __device__ inline bool IsFirstProcessor() const { return up < 0; }
    __host__ __device__ inline bool IsLastProcessor() const { return down < 0; }
    __host__ __device__ inline bool IsRightProcessor() const { return right < 0; }
    __host__ __device__ inline bool IsLeftProcessor() const { return left < 0; }

    __host__ __device__ inline int RowsCount() const { return rowsCountValue; }
    __host__ __device__ inline int ColsCount() const { return colsCountValue; }
    __host__ __device__ inline int RowsCountWithBorders() const { return rowsCountWithBorders; }
    __host__ __device__ inline int ColsCountWithBorders() const { return colsCountWithBorders; }

    __host__ __device__ inline int FirstBorderRowRelativeIndex () const { return IsFirstProcessor() ? 1 : -1; }
    __host__ __device__ inline int LastBorderRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 2 : -1; }
    __host__ __device__ inline int FirstBorderColRelativeIndex () const { return IsLeftProcessor() ? 1 : -1; }
    __host__ __device__ inline int LastBorderColRelativeIndex () const { return IsRightProcessor() ? ColsCountWithBorders() - 2 : -1; }

    inline int FirstRowIndex() const { return startRowIndex; }
    inline int LastRowIndex() const { return startRowIndex + rowsCountValue - 1; }
    inline int FirstColIndex() const { return startColIndex; }
    inline int LastColIndex() const { return startColIndex + colsCountValue - 1; }

    __host__ __device__ inline int FirstInnerRowRelativeIndex () const { return IsFirstProcessor() ? 2 : 1; }
    __host__ __device__ inline int LastInnerRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 3 : RowsCountWithBorders() - 2; }
    __host__ __device__ inline int FirstInnerColRelativeIndex () const { return IsLeftProcessor() ? 2 : 1; }
    __host__ __device__ inline int LastInnerColRelativeIndex () const { return IsRightProcessor() ? ColsCountWithBorders() - 3 : ColsCountWithBorders() - 2; }

    __host__ __device__ inline int FirstOwnRowRelativeIndex () const { return 1; }
    __host__ __device__ inline int LastOwnRowRelativeIndex () const { return RowsCountWithBorders() - 2; }
    __host__ __device__ inline int FirstOwnColRelativeIndex () const { return 1; }
    __host__ __device__ inline int LastOwnColRelativeIndex () const { return ColsCountWithBorders() - 2; }


    __device__ bool IsBorderIndices(int i, int j) const
    {
        bool isRowBorder = (i == FirstBorderRowRelativeIndex() || i == LastBorderRowRelativeIndex()) &&
            j >= FirstOwnColRelativeIndex() && j <= LastOwnColRelativeIndex();
        bool isColBorder = (j == FirstBorderColRelativeIndex() || j == LastBorderColRelativeIndex()) &&
            i >= FirstOwnRowRelativeIndex() && i <= LastOwnRowRelativeIndex();
        return isRowBorder || isColBorder;
    }


    void InitCartParameters(int p0, int p1, int N0Value, int N1Value)
    {

        dims[0] = (unsigned int) 1 << p0;   dims[1] = (unsigned int) 1 << p1;
        N0 = N0Value;                       N1 = N1Value;
        n0 = N0 >> p0;                      n1 = N1 >> p1;
        k0 = N0 - dims[0]*n0;               k1 = N1 - dims[1]*n1;
    }

    void InitProcessorRowsParameters()
    {
        if (k1 == 0 || (k1 > 0 && jCartIndex >= k1))
        {
            startRowIndex  = k1 * (n1 + 1) + (jCartIndex - k1) * n1;
            rowsCountValue = n1;
        }
        else // k1 > 0, last coubs
        {
            startRowIndex  = jCartIndex * (n1 + 1);
            rowsCountValue = n1 + 1;
        }        
        rowsCountWithBorders = rowsCountValue + 2;
        // reverse, as we want matrix to start in left up corner
        int lastRowIndex = startRowIndex  + rowsCountValue - 1;
        startRowIndex = N1 - lastRowIndex - 1;
        return;
    }

    void InitProcessorColsParameters()
    {
        if (k0 == 0 || (k0 > 0 && iCartIndex >= k0))
        {
            startColIndex = k0 * (n0 + 1) + (iCartIndex - k0) * n0;
            colsCountValue = n0;
        }
        else // k0 > 0, last coubs
        {
            startColIndex = iCartIndex * (n0 + 1);
            colsCountValue = n0 + 1;
        }
        colsCountWithBorders = colsCountValue + 2;
        return;
    }

    void InitCudaParameters();


    dim3 GetThreadsPerBlocks2Dim(int numCols, int numRows) const
    {
        int maxThreadsPerBlock2Dim = sqrt(devProp.maxThreadsPerBlock) / 2;
        int threadsPerBlockX = std::min(numCols, maxThreadsPerBlock2Dim);
        int threadsPerBlockY = std::min(numRows, maxThreadsPerBlock2Dim);
        dim3 threadsPerBlock(threadsPerBlockX, threadsPerBlockY);
        return threadsPerBlock;
    }

    dim3 GetBlocksPerGrid2Dim(dim3 threadsPerBlock, int numCols, int numRows) const
    {
        dim3 numBlocks((numCols - 1) / threadsPerBlock.x + 1,
                       (numRows - 1) / threadsPerBlock.y + 1);
        return numBlocks;
    }

    dim3 GetThreadsPerBlocks1Dim(int numObjects) const
    {
        int maxThreadsPerBlock1Dim = devProp.maxThreadsPerBlock / 2;
        int threadsPerBlockX = std::min(numObjects, maxThreadsPerBlock1Dim);
        dim3 threadsPerBlock(threadsPerBlockX);
        return threadsPerBlock;
    }

    dim3 GetBlocksPerGrid1Dim(dim3 threadsPerBlock, int numObjects) const
    {
         dim3 numBlocks((numObjects - 1) / threadsPerBlock.x + 1);
         return numBlocks;
    }


};

#endif // PROCESSORSDATA_H
