#ifndef PROCESSORSDATA_H
#define PROCESSORSDATA_H

#include "interface.h"
#include <mpi.h>

class ProcessorsData
{
public:
    const int mainProcessorRank = 0;
    int rank, processorsCount;
    int iCartIndex, jCartIndex;
    int startRowIndex, endRowIndex, rowsCountValue;
    int startColIndex, endColIndex, colsCountValue;
    int left, right, down, up;
    int n1, k1, n0, k0, N0, N1;
    MPI_Comm gridComm, rowComm, colComm;

    ProcessorsData(int rankValue, int processorsCountValue,
                   int leftIndex, int rightIndex,
                   int downIndex, int upIndex):
        rank(rankValue), processorsCount(processorsCountValue),
        left(leftIndex), right(rightIndex),
        down(downIndex), up(upIndex) { }

    inline bool IsMainProcessor() const { return rank == mainProcessorRank; }
    inline bool IsFirstProcessor() const { return up == -1; }
    inline bool IsLastProcessor() const { return down == -1; }
    inline bool IsRightProcessor() const { return right == -1; }
    inline bool IsLeftProcessor() const { return left == -1; }

    inline int RowsCount() const { return rowsCountValue; }
    inline int ColsCount() const { return colsCountValue; }
    inline int RowsCountWithBorders() const { return rowsCountValue + 2; }
    inline int ColsCountWithBorders() const { return colsCountValue + 2; }

    inline int FirstRowIndex() const { return startRowIndex; }
    inline int LastRowIndex() const { return startRowIndex + rowsCountValue - 1; }
    inline int FirstColIndex() const { return startColIndex; }
    inline int LastColIndex() const { return startColIndex + colsCountValue - 1; }

    inline int FirstRowWithBordersIndex() const { return startRowIndex - 1 >= 0 ? startRowIndex - 1 : 0; }
    inline int LastRowWithBordersIndex() const { return IsLastProcessor() ? LastRowIndex() : startRowIndex + rowsCountValue + 2 - 1; }

    inline int FirstInnerRowRelativeIndex () const { return IsFirstProcessor() ? 2 : 1; }
    inline int LastInnerRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 3 : RowsCountWithBorders() - 2; }
    inline int FirstInnerColRelativeIndex () const { return IsLeftProcessor() ? 2 : 1; }
    inline int LastInnerColRelativeIndex () const { return IsRightProcessor() ? ColsCountWithBorders() - 3 : ColsCountWithBorders() - 2; }


    inline int FirstBorderRowRelativeIndex () const { return IsFirstProcessor() ? 1 : -1; }
    inline int LastBorderRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 2 : -1; }
    inline int FirstBorderColRelativeIndex () const { return IsLeftProcessor() ? 1 : -1; }
    inline int LastBorderColRelativeIndex () const { return IsRightProcessor() ? ColsCountWithBorders() - 2 : -1; }

    inline bool IsInnerIndices(int i, int j) const
    {
        return i == FirstBorderRowRelativeIndex() || i == LastBorderRowRelativeIndex() ||
               j == FirstBorderColRelativeIndex() || j == LastBorderColRelativeIndex();
    }

    inline int FirstOwnRowRelativeIndex () const { return 1; }
    inline int LastOwnRowRelativeIndex () const { return RowsCountWithBorders() - 2; }
    inline int FirstOwnColRelativeIndex () const { return 1; }
    inline int LastOwnColRelativeIndex () const { return ColsCountWithBorders() - 2; }

    void InitRowsParameters(std::pair<int, int> rowsParams)
    {
        rowsCountValue = rowsParams.first;
        startRowIndex = rowsParams.second;
    }

    void InitColsParameters(std::pair<int, int> colsParams)
    {
        colsCountValue = colsParams.first;
        startColIndex = colsParams.second;
    }

    void InitCartParameters(int n0Value, int k0Value, int n1Value, int k1Value, int N0Value, int N1Value)
    {
        n0 = n0Value; n1 = n1Value;
        k0 = k0Value; k1 = k1Value;
        N0 = N0Value; N1 = N1Value;
    }

    void InitCartCoordinates(int i, int j)
    {
        iCartIndex = i; jCartIndex = j;
    }

    void InitComms(MPI_Comm gridCommValue, MPI_Comm rowCommValue, MPI_Comm colCommValue)
    {
        gridComm  = gridCommValue; rowComm = rowCommValue; colComm = colCommValue;
    }

    std::pair<int, int> static GetProcessorParameters(int pointsCount, int rankValue, int processorsCount)
    {
        std::cerr << "err" << std::endl;
        return std::make_pair(-1, -1);
    }


    std::pair<int, int> static GetProcessorRowsParameters(int N1, int n1, int k1, int jProcessorIndex)
    {
        int rowsCount, firstRowIndex;
        if (k1 == 0 || (k1 > 0 && jProcessorIndex >= k1))
        {
            firstRowIndex = k1 * (n1 + 1) + (jProcessorIndex - k1) * n1;
            rowsCount = n1;
        }
        else // k1 > 0, last coubs
        {
            firstRowIndex = jProcessorIndex * (n1 + 1);
            rowsCount = n1 + 1;
        }
        // reverse, as we want matrix to start in left up corner
        auto lastRowIndex = firstRowIndex + rowsCount - 1;
        firstRowIndex = N1 - lastRowIndex - 1;
        return std::make_pair(rowsCount, firstRowIndex);
    }

    std::pair<int, int> static GetProcessorColsParameters(int n0, int k0, int iProcessorIndex)
    {
        int colsCount, firstColIndex;
        if (k0 == 0 || (k0 > 0 && iProcessorIndex >= k0))
        {
            firstColIndex = k0 * (n0 + 1) + (iProcessorIndex - k0) * n0;
            colsCount = n0;
        }
        else // k0 > 0, last coubs
        {
            firstColIndex = iProcessorIndex * (n0 + 1);
            colsCount = n0 + 1;
        }
        return std::make_pair(colsCount, firstColIndex);
    }
};

#endif // PROCESSORSDATA_H
