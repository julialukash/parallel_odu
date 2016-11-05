#ifndef PROCESSORSDATA_H
#define PROCESSORSDATA_H

#include "interface.h"

class ProcessorsData
{
public:
    const int mainProcessorRank = 0;
    int rank, processorsCount;
    int startRowIndex, endRowIndex, rowsCountValue;
    int startColIndex, endColIndex, colsCountValue;
    int left, right, down, up;

    ProcessorsData(int rankValue, int processorsCountValue,
                   int leftIndex, int rightIndex,
                   int downIndex, int upIndex):
        rank(rankValue), processorsCount(processorsCountValue),
        left(leftIndex), right(rightIndex),
        down(downIndex), up(upIndex) { }

    inline bool IsMainProcessor() const { return rank == mainProcessorRank; }
    inline bool IsFirstProcessor() const { return rank == 0; }
    inline bool IsLastProcessor() const { return rank == processorsCount - 1; }

    inline int RowsCount() const { return rowsCountValue; }
    inline int RowsCountWithBorders() const { return rowsCountValue + 2; }

    inline int FirstRowIndex() const { return startRowIndex; }
    inline int LastRowIndex() const { return startRowIndex + rowsCountValue - 1; }

    inline int FirstRowWithBordersIndex() const { return startRowIndex - 1 >= 0 ? startRowIndex - 1 : 0; }
    inline int LastRowWithBordersIndex() const { return IsLastProcessor() ? LastRowIndex() : startRowIndex + rowsCountValue + 2 - 1; }


    inline int FirstRowRelativeIndex () const { return IsFirstProcessor() ? 2 : 1; }
    inline int LastRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 3 : RowsCountWithBorders() - 2; }

    std::pair<int, int> static GetProcessorParameters(int pointsCount, int rankValue, int processorsCount)
    {
        std::cerr << "err" << std::endl;
        return std::make_pair(-1, -1);
    }

    std::pair<int, int> static GetProcessorRowsParameters(int n1, int k1, int jProcessorIndex)
    {
        int rowsCount, firstRowIndex;
        if (k1 == 0 || k1 > 0 && jProcessorIndex >= k1)
        {
            firstRowIndex = k1 * (n1 + 1) + (jProcessorIndex - k1) * n1;
            rowsCount = n1;
        }
        else // k1 > 0, last coubs
        {
            firstRowIndex = jProcessorIndex * (n1 + 1);
            rowsCount = n1 + 1;
        }
        return std::make_pair(rowsCount, firstRowIndex);
    }

    std::pair<int, int> static GetProcessorColsParameters(int n0, int k0, int iProcessorIndex)
    {
        int colsCount, firstColIndex;
        if (k0 == 0 || k0 > 0 && iProcessorIndex >= k0)
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
