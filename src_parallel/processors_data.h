#ifndef PROCESSORSDATA_H
#define PROCESSORSDATA_H

#include "interface.h"
#include <mpi.h>

class ProcessorsData
{
private:
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
        return(p0);
    }

public:
    const int mainProcessorRank = 0;
    int rank, processorsCount;
    int iCartIndex, jCartIndex;
    int startRowIndex, endRowIndex, rowsCountValue;
    int startColIndex, endColIndex, colsCountValue;
    int left, right, down, up;
    int n1, k1, n0, k0, N0, N1;
    int dims[2];
    MPI_Comm gridComm;

    ProcessorsData(int processorsCountValue): processorsCount(processorsCountValue){ }

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

    void InitCartParameters(int power, int N0Value, int N1Value)
    {
        int p0 = SplitFunction(N0Value, N1Value, power);
        int p1 = power - p0;

        dims[0] = (unsigned int) 1 << p0;   dims[1] = (unsigned int) 1 << p1;
        N0 = N0Value;                       N1 = N1Value;
        n0 = N0 >> p0;                      n1 = N1 >> p1;
        k0 = N0 - dims[0]*n0;               k1 = N1 - dims[1]*n1;
    }

    void InitCartCoordinates(int i, int j)
    {
        iCartIndex = i; jCartIndex = j;
    }

    void InitComms(MPI_Comm gridCommValue)
    {
        gridComm  = gridCommValue;
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
        auto lastRowIndex = startRowIndex  + rowsCountValue - 1;
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
