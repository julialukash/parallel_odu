#ifndef PROCESSORSDATA_H
#define PROCESSORSDATA_H

#include "Interface.h"

class ProcessorsData
{
public:
    const int mainProcessorRank = 0;
    int rank, processorsCount;
    int startRowIndex, endRowIndex, rowsCountValue;
    int yPoints;

    DoubleMatrix u, p, previousP, grad, laplassGrad, laplassPreviousGrad;

    ProcessorsData(int rankValue, int processorsCountValue)
    {
        rank = rankValue;
        processorsCount = processorsCountValue;
    }

    inline bool IsMainProcessor() { return rank == mainProcessorRank; }
    inline bool IsLastProcessor() { return rank == processorsCount - 1; }

    inline int RowsCount() { return rowsCountValue; }
    inline int RowsCountWithBorders() { return rowsCountValue + 2; }

    inline int FirstRowIndex() { return startRowIndex; }
    inline int LastRowIndex() { return startRowIndex + rowsCountValue - 1; }

    inline int FirstRowWithBordersIndex() { return startRowIndex - 1 >= 0 ? startRowIndex - 1 : 0; }
    inline int LastRowWithBordersIndex() { return IsLastProcessor() ? LastRowIndex() : startRowIndex + rowsCountValue + 2 - 1; }
};

#endif // PROCESSORSDATA_H
