#ifndef PROCESSORSDATA_H
#define PROCESSORSDATA_H

#include "interface.h"

class ProcessorsData
{
public:
    const int mainProcessorRank = 0;
    int rank, processorsCount;
    int startRowIndex, endRowIndex, rowsCountValue;
    int yPoints;

    ProcessorsData(int rankValue, int processorsCountValue, int yPointsValue)
    {
        rank = rankValue;
        processorsCount = processorsCountValue;
        yPoints = yPointsValue;
    }

    inline bool IsMainProcessor() const { return rank == mainProcessorRank; }
    inline bool IsFirstProcessor() const { return rank == 0; }
    inline bool IsLastProcessor() const { return rank == processorsCount - 1; }

    inline int RowsCount() const { return rowsCountValue; }
    inline int ColsCount() const { return yPoints; }
    inline int RowsCountWithBorders() const { return rowsCountValue + 2; }

    inline int FirstRowIndex() const { return startRowIndex; }
    inline int LastRowIndex() const { return startRowIndex + rowsCountValue - 1; }

    inline int FirstColIndex() const { return 0; }
    inline int LastColIndex() const { return yPoints - 1; }

    inline int FirstRowWithBordersIndex() const { return startRowIndex - 1 >= 0 ? startRowIndex - 1 : 0; }
    inline int LastRowWithBordersIndex() const { return IsLastProcessor() ? LastRowIndex() : startRowIndex + rowsCountValue + 2 - 1; }

    inline int FirstRowRelativeIndex () const { return IsFirstProcessor() ? 2 : 1; }
    inline int LastRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 3 : RowsCountWithBorders() - 2; }

    std::pair<int, int> static GetProcessorParameters(int pointsCount, int rankValue, int processorsCount)
    {
        int rowsCount, firstRowIndex;
        rowsCount = pointsCount / processorsCount;
        auto leftRowsCount = pointsCount - rowsCount * processorsCount;
        if (rankValue < leftRowsCount)
        {
            rowsCount = rowsCount + 1;
            firstRowIndex = rankValue * rowsCount;
        }
        else
        {
            firstRowIndex = leftRowsCount * (rowsCount + 1) + (rankValue - leftRowsCount) * rowsCount;
        }
        return std::make_pair(rowsCount, firstRowIndex);
    }

};

#endif // PROCESSORSDATA_H
