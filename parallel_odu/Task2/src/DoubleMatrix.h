#ifndef DOUBLEMATRIX_H
#define DOUBLEMATRIX_H

#include <iostream>

class DoubleMatrix
{
private:
    int rowsCount, colsCount;

public:
    double **matrix;

    DoubleMatrix()
    {
        matrix = NULL;
        rowsCount = 0;
        colsCount = 0;
    }

    ~DoubleMatrix()
    {
        for (int r = 0; r < rowsCount; r++)
        {
            delete matrix[r];
        }
        delete matrix;
        matrix = NULL;
    }

    DoubleMatrix(const int rowCount, const int colCount)
    {
      matrix = NULL;
      rowsCount = rowCount;
      colsCount = colCount;

      matrix = new double*[rowsCount];
      for (auto i = 0; i < rowsCount; i++)
      {
          matrix[i] = new double[colsCount];
          for (auto j = 0; j < colsCount; j++)
          {
              matrix[i][j] = 0;
          }
      }
    }

    DoubleMatrix(const DoubleMatrix& otherMatrix)
    {
        rowsCount = otherMatrix.rowsCount;
        colsCount = otherMatrix.colsCount;
        matrix = new double*[otherMatrix.rowsCount];
        for (auto i = 0; i < rowsCount; i++)
        {
            matrix[i] = new double[colsCount];
            for (auto j = 0; j < colsCount; j++)
            {
                matrix[i][j] = otherMatrix.matrix[i][j];
            }
        }
     }

    double& operator()(const int i, const int j) const
    {
        if (matrix != NULL && i >= 0 && i < rowsCount && j >= 0 && j < colsCount)
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
        rowsCount = otherMatrix.rowsCount;
        colsCount = otherMatrix.colsCount;
        matrix = new double*[otherMatrix.rowsCount];
        for (auto i = 0; i < rowsCount; i++)
        {
            matrix[i] = new double[colsCount];
            for (auto j = 0; j < colsCount; j++)
            {
                matrix[i][j] = otherMatrix.matrix[i][j];
            }
        }
        return *this;
    }

    friend DoubleMatrix operator+(const DoubleMatrix& a, const DoubleMatrix& b)
    {
        if (a.rowsCount == b.rowsCount && a.colsCount == b.colsCount)
        {
            DoubleMatrix res(a.rowsCount, a.colsCount);

            for (auto i = 0; i < a.rowsCount; i++)
            {
                for (auto j = 0; j < a.colsCount; j++)
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
        if (a.rowsCount == b.rowsCount && a.colsCount == b.colsCount)
        {
            DoubleMatrix res(a.rowsCount, a.colsCount);

            for (auto i = 0; i < a.rowsCount; i++)
            {
                for (auto j = 0; j < a.colsCount; j++)
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
        for (auto i = 0; i < rowsCount; i++)
        {
            for (auto j = 0; j < colsCount; j++)
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

    int size1() const
    {
        return rowsCount;
    }

    int size2() const
    {
        return colsCount;
    }


    friend std::ostream& operator<<(std::ostream& os, const DoubleMatrix& dt)
    {
        for (auto i = 0; i < dt.size1(); ++i)
        {
            for (auto j = 0; j < dt.size2(); ++j)
            {
                os << dt(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }

};


#endif // DOUBLEMATRIX_H
