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

    inline int FirstInnerRowRelativeIndex () const { return IsFirstProcessor() ? 2 : 1; }
    inline int LastInnerRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 3 : RowsCountWithBorders() - 2; }
    inline int FirstInnerColRelativeIndex () const { return 1; }
    inline int LastInnerColRelativeIndex () const { return ColsCount() - 2; }


    inline int FirstBorderRowRelativeIndex () const { return IsFirstProcessor() ? 1 : -1; }
    inline int LastBorderRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 2 : -1; }
    inline int FirstBorderColRelativeIndex () const { return 0; }
    inline int LastBorderColRelativeIndex () const { return ColsCount() - 1; }

    inline bool IsInnerIndices(int i, int j) const
    {
        return i == FirstBorderRowRelativeIndex() || i == LastBorderRowRelativeIndex() ||
               j == FirstBorderColRelativeIndex() || j == LastBorderColRelativeIndex();
    }

    inline int FirstOwnRowRelativeIndex () const { return 1; }
    inline int LastOwnRowRelativeIndex () const { return RowsCountWithBorders() - 2; }
    inline int FirstOwnColRelativeIndex () const { return 0; }
    inline int LastOwnColRelativeIndex () const { return ColsCount() - 1; }



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
