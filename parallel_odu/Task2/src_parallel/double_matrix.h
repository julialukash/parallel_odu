#ifndef DOUBLEMATRIX_H
#define DOUBLEMATRIX_H


#include <iostream>
#include <memory.h>

#define PARALLEL_M

class DoubleMatrix
{
private:
public:
    int rowsCountValue, colsCountValue;
    double *matrix;

    ~DoubleMatrix()
    {
        delete[] matrix;
    }

    DoubleMatrix(const int rowCount, const int colCount)
    {
      rowsCountValue = rowCount;
      colsCountValue = colCount;

      matrix = new double[rowsCountValue * colsCountValue];
#ifdef PARALLEL_M
      #pragma omp parallel for
#endif
      for (int i = 0; i < rowsCountValue; i++)
      {
          for (int j = 0; j < colsCountValue; j++)
          {
              matrix[i * colsCountValue + j] = 0;
          }
      }
    }

    DoubleMatrix(const DoubleMatrix& otherMatrix)
    {
        rowsCountValue = otherMatrix.rowsCountValue;
        colsCountValue = otherMatrix.colsCountValue;
        matrix = new double[otherMatrix.rowsCountValue * otherMatrix.colsCountValue];
#ifdef PARALLEL_M
      #pragma omp parallel for
#endif
        for (int i = 0; i < rowsCountValue; i++)
        {
            for (int j = 0; j < colsCountValue; j++)
            {
                operator() (i, j) = otherMatrix(i, j);
            }
        }
     }

    double& operator()(const int i, const int j) const
    {
        if (matrix != NULL && i >= 0 && i < rowsCountValue && j >= 0 && j < colsCountValue)
        {
          return matrix[i * colsCountValue + j];
        }
        else
        {
          throw "Subscript out of range";
        }
    }

    double& operator[](const int i) const
    {
        if (matrix != NULL && i >= 0 && i < rowsCountValue)
        {
          return matrix[i * colsCountValue];
        }
        else
        {
          throw "Subscript out of range";
        }
    }

    friend DoubleMatrix* operator-(const DoubleMatrix& a, const DoubleMatrix& b)
    {
        if (a.rowsCountValue == b.rowsCountValue && a.colsCountValue == b.colsCountValue)
        {
            DoubleMatrix* res = new DoubleMatrix(a.rowsCountValue, a.colsCountValue);

#ifdef PARALLEL_M
      #pragma omp parallel for
#endif
            for (int i = 0; i < a.rowsCountValue; i++)
            {
                for (int j = 0; j < a.colsCountValue; j++)
                {
                    (*res)(i, j) = a(i, j) - b(i, j);
                }
            }
            return res;
        }
        else
        {
            throw "Dimensions does not match";
        }
    }

    friend DoubleMatrix* operator* (const double b, const DoubleMatrix & a)
    {
        DoubleMatrix* res = new DoubleMatrix(a.rowsCount(), a.colsCount());
#ifdef PARALLEL_M
      #pragma omp parallel for
#endif
        for (int i = 0; i < res->rowsCount(); i++)
        {
            for (int j = 0; j < res->colsCount(); j++)
            {
                (*res)(i, j) = a(i,j) * b;
            }
        }
        return res;
    }

    int rowsCount() const
    {
        return rowsCountValue;
    }

    int colsCount() const
    {
        return colsCountValue;
    }

    DoubleMatrix* CropMatrix(int startRow, int rowsCount, int startCol, int colsCount) const
    {
        DoubleMatrix* res = new DoubleMatrix(rowsCount, colsCount);
#ifdef PARALLEL_M
      #pragma omp parallel for
#endif
        for (int i = startRow; i < startRow + rowsCount; i++)
        {
            int iIndex = i - startRow;
            for (int j = startCol; j < startCol + colsCount; j++)
            {
                int jIndex = j - startCol;
                (*res)(iIndex, jIndex) = operator ()(i, j);
            }
        }
        return res;
    }

    void SetNewColumn(const DoubleMatrix& column, int columnIndex)
    {
        if (column.rowsCount() != rowsCount() || column.colsCount() != 1 || columnIndex >= colsCount())
        {
            std::cerr << "Incorrect column\n" << std::endl;
            throw "Incorrect column\n";
        }
#ifdef PARALLEL_M
      #pragma omp parallel for
#endif
        for (int i = 0; i < rowsCount(); ++i)
        {
            operator() (i, columnIndex) = column(i, 0);
        }
    }


    friend std::ostream& operator<<(std::ostream& os, const DoubleMatrix& dt)
    {
        for (int i = 0; i < dt.rowsCount(); ++i)
        {
            for (int j = 0; j < dt.colsCount(); ++j)
            {
                os << dt(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }
};

#endif // DOUBLEMATRIX_H
