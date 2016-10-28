#ifndef DOUBLEMATRIX_H
#define DOUBLEMATRIX_H


#include <iostream>

class DoubleMatrix
{
private:
    int rowsCountValue, colsCountValue;
public:
    double **matrix;

    DoubleMatrix()
    {
        matrix = NULL;
        rowsCountValue = 0;
        colsCountValue = 0;
    }


    DoubleMatrix(const int rowCount, const int colCount)
    {
      matrix = NULL;
      rowsCountValue = rowCount;
      colsCountValue = colCount;

      matrix = new double*[rowsCountValue];
      for (auto i = 0; i < rowsCountValue; i++)
      {
          matrix[i] = new double[colsCountValue];
          for (auto j = 0; j < colsCountValue; j++)
          {
              matrix[i][j] = 0;
          }
      }
    }

    DoubleMatrix(const DoubleMatrix& otherMatrix)
    {
        rowsCountValue = otherMatrix.rowsCountValue;
        colsCountValue = otherMatrix.colsCountValue;
        matrix = new double*[otherMatrix.rowsCountValue];
        for (auto i = 0; i < rowsCountValue; i++)
        {
            matrix[i] = new double[colsCountValue];
            for (auto j = 0; j < colsCountValue; j++)
            {
                matrix[i][j] = otherMatrix.matrix[i][j];
            }
        }
     }

    const double* getRow(const int i) const
    {
        return matrix[i];
    }


    double* getRowNotConst(const int i)
    {
        return matrix[i];
    }

    double& operator()(const int i, const int j) const
    {
        if (matrix != NULL && i >= 0 && i < rowsCountValue && j >= 0 && j < colsCountValue)
        {
          return matrix[i][j];
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
        matrix = new double*[otherMatrix.rowsCountValue];
        for (auto i = 0; i < rowsCountValue; i++)
        {
            matrix[i] = new double[colsCountValue];
            for (auto j = 0; j < colsCountValue; j++)
            {
                matrix[i][j] = otherMatrix.matrix[i][j];
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
                    res.matrix[i][j] = a.matrix[i][j] + b.matrix[i][j];
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
                    res.matrix[i][j] = a.matrix[i][j] - b.matrix[i][j];
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
                matrix[i][j] *= value;
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

    DoubleMatrix CropMatrix(const DoubleMatrix& a, int startRow, int rowsCount) const
    {
        DoubleMatrix res(rowsCount, a.colsCountValue);
        for (auto i = startRow; i < startRow + rowsCount; i++)
        {
            auto iIndex = i - startRow;
            for (auto j = 0; j < a.colsCountValue; j++)
            {
                res.matrix[iIndex][j] = a.matrix[i][j];
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
