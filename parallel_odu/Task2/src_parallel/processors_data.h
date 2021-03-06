#ifndef PROCESSORSDATA_H
#define PROCESSORSDATA_H

#include "interface.h"
#include <mpi.h>

class ProcessorsData
{
private:
public:
    int rank, processorsCount;
    int iCartIndex, jCartIndex;
    int startRowIndex, endRowIndex, rowsCountValue;
    int startColIndex, endColIndex, colsCountValue;
    int left, right, down, up;
    int n1, k1, n0, k0, N0, N1;
    int dims[2];
    MPI_Comm gridComm;

    ProcessorsData(int processorsCountValue): processorsCount(processorsCountValue){ }

    inline bool IsMainProcessor() const { return rank == 0; }
    inline bool IsFirstProcessor() const { return up < 0; }
    inline bool IsLastProcessor() const { return down < 0; }
    inline bool IsRightProcessor() const { return right < 0; }
    inline bool IsLeftProcessor() const { return left < 0; }

    inline int RowsCount() const { return rowsCountValue; }
    inline int ColsCount() const { return colsCountValue; }
    inline int RowsCountWithBorders() const { return rowsCountValue + 2; }
    inline int ColsCountWithBorders() const { return colsCountValue + 2; }

    inline int FirstRowIndex() const { return startRowIndex; }
    inline int LastRowIndex() const { return startRowIndex + rowsCountValue - 1; }
    inline int FirstColIndex() const { return startColIndex; }
    inline int LastColIndex() const { return startColIndex + colsCountValue - 1; }

    inline int FirstInnerRowRelativeIndex () const { return IsFirstProcessor() ? 2 : 1; }
    inline int LastInnerRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 3 : RowsCountWithBorders() - 2; }
    inline int FirstInnerColRelativeIndex () const { return IsLeftProcessor() ? 2 : 1; }
    inline int LastInnerColRelativeIndex () const { return IsRightProcessor() ? ColsCountWithBorders() - 3 : ColsCountWithBorders() - 2; }


    inline int FirstBorderRowRelativeIndex () const { return IsFirstProcessor() ? 1 : -1; }
    inline int LastBorderRowRelativeIndex () const { return IsLastProcessor() ? RowsCountWithBorders() - 2 : -1; }
    inline int FirstBorderColRelativeIndex () const { return IsLeftProcessor() ? 1 : -1; }
    inline int LastBorderColRelativeIndex () const { return IsRightProcessor() ? ColsCountWithBorders() - 2 : -1; }

    inline int FirstOwnRowRelativeIndex () const { return 1; }
    inline int LastOwnRowRelativeIndex () const { return RowsCountWithBorders() - 2; }
    inline int FirstOwnColRelativeIndex () const { return 1; }
    inline int LastOwnColRelativeIndex () const { return ColsCountWithBorders() - 2; }

    inline bool IsInnerIndices(int i, int j) const
    {
        return i == FirstBorderRowRelativeIndex() || i == LastBorderRowRelativeIndex() ||
               j == FirstBorderColRelativeIndex() || j == LastBorderColRelativeIndex();
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
        return;
    }
};

#endif // PROCESSORSDATA_H
