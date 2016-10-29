#ifndef DOUBLEMATRIX_H
#define DOUBLEMATRIX_H


#include <iostream>

class DoubleMatrix
{
private:
    int rowsCountValue, colsCountValue;
public:
    double *matrix;

    DoubleMatrix()
    {
        matrix = NULL;
        rowsCountValue = 0;
        colsCountValue = 0;
    }

    ~DoubleMatrix()
    {
        delete[] matrix;
    }

    DoubleMatrix(const int rowCount, const int colCount)
    {
      matrix = NULL;
      rowsCountValue = rowCount;
      colsCountValue = colCount;

      matrix = new double[rowsCountValue * colsCountValue];
      for (auto i = 0; i < rowsCountValue; i++)
      {
          for (auto j = 0; j < colsCountValue; j++)
          {
              matrix[i * rowsCountValue + j] = 0;
          }
      }
    }

    DoubleMatrix(const DoubleMatrix& otherMatrix)
    {
        rowsCountValue = otherMatrix.rowsCountValue;
        colsCountValue = otherMatrix.colsCountValue;
        matrix = new double[otherMatrix.rowsCountValue * otherMatrix.colsCountValue];
        for (auto i = 0; i < rowsCountValue; i++)
        {
            for (auto j = 0; j < colsCountValue; j++)
            {
                operator() (i, j) = otherMatrix(i, j);
            }
        }
     }


    DoubleMatrix(double* otherMatrix, int rowsCount, int colsCount)
    {
        rowsCountValue = rowsCount;
        colsCountValue = colsCount;
        matrix = new double[rowsCount * colsCount];
        for (auto i = 0; i < rowsCountValue; i++)
        {
            for (auto j = 0; j < colsCountValue; j++)
            {
                operator() (i, j) = otherMatrix[i * colsCount + j];
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


    // assignment operator
    DoubleMatrix& operator= (const DoubleMatrix& otherMatrix)
    {
        rowsCountValue = otherMatrix.rowsCountValue;
        colsCountValue = otherMatrix.colsCountValue;
        matrix = new double[otherMatrix.rowsCountValue * otherMatrix.colsCountValue];
        for (auto i = 0; i < rowsCountValue; i++)
        {
            for (auto j = 0; j < colsCountValue; j++)
            {
                operator() (i, j) = otherMatrix(i, j);
            }
        }
        return *this;
    }

    friend DoubleMatrix operator+(const DoubleMatrix& a, const DoubleMatrix& b)
    {
        if (a.rowsCountValue == b.rowsCountValue && a.colsCountValue == b.colsCountValue)
        {
            DoubleMatrix res(a.rowsCountValue, a.colsCountValue);

            for (auto i = 0; i < a.rowsCountValue; i++)
            {
                for (auto j = 0; j < a.colsCountValue; j++)
                {
                    res(i, j) = a(i, j) + b(i, j);
                }
            }
            return res;
        }
        else
        {
            throw "Dimensions does not match";
        }
    }

    friend DoubleMatrix operator-(const DoubleMatrix& a, const DoubleMatrix& b)
    {
        if (a.rowsCountValue == b.rowsCountValue && a.colsCountValue == b.colsCountValue)
        {
            DoubleMatrix res(a.rowsCountValue, a.colsCountValue);

            for (auto i = 0; i < a.rowsCountValue; i++)
            {
                for (auto j = 0; j < a.colsCountValue; j++)
                {
                    res(i, j) = a(i, j) - b(i, j);
                }
            }
            return res;
        }
        else
        {
            throw "Dimensions does not match";
        }
    }

    DoubleMatrix& MultiplyByValue(const double value)
    {
        for (auto i = 0; i < rowsCountValue; i++)
        {
            for (auto j = 0; j < colsCountValue; j++)
            {
                operator()(i, j) *= value;
            }
        }
        return *this;
    }

    friend DoubleMatrix operator* (const DoubleMatrix & a, const double b)
    {
        DoubleMatrix res = a;
        res.MultiplyByValue(b);
        return res;
    }

    friend DoubleMatrix operator* (const double b, const DoubleMatrix & a)
    {
        DoubleMatrix res = a;
        res.MultiplyByValue(b);
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

    DoubleMatrix CropMatrix(int startRow, int rowsCount) const
    {
        DoubleMatrix res(rowsCount, colsCountValue);
        for (auto i = startRow; i < startRow + rowsCount; i++)
        {
            auto iIndex = i - startRow;
            for (auto j = 0; j < colsCountValue; j++)
            {
                res(iIndex, j) = operator ()(i, j);
            }
        }
        return res;
    }


    friend std::ostream& operator<<(std::ostream& os, const DoubleMatrix& dt)
    {
        for (auto i = 0; i < dt.rowsCount(); ++i)
        {
            for (auto j = 0; j < dt.colsCount(); ++j)
            {
                os << dt(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }
};

#endif // DOUBLEMATRIX_H
