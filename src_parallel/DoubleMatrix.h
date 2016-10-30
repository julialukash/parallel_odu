#ifndef DOUBLEMATRIX_H
#define DOUBLEMATRIX_H


#include <iostream>
#include <memory.h>

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
//        std::cout << "BITCH, I AM COPYING " << std::endl;
        rowsCountValue = otherMatrix.rowsCountValue;
        colsCountValue = otherMatrix.colsCountValue;
        matrix = new double[otherMatrix.rowsCountValue * otherMatrix.colsCountValue];
        for (int i = 0; i < rowsCountValue; i++)
        {
            for (int j = 0; j < colsCountValue; j++)
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
        for (int i = 0; i < rowsCountValue; i++)
        {
            for (int j = 0; j < colsCountValue; j++)
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
        for (int i = 0; i < rowsCountValue; i++)
        {
            for (int j = 0; j < colsCountValue; j++)
            {
                operator() (i, j) = otherMatrix(i, j);
            }
        }
        return *this;
    }


    friend std::shared_ptr<DoubleMatrix> operator-(const DoubleMatrix& a, const DoubleMatrix& b)
    {
        if (a.rowsCountValue == b.rowsCountValue && a.colsCountValue == b.colsCountValue)
        {
            auto res = std::shared_ptr<DoubleMatrix>(new DoubleMatrix(a.rowsCountValue, a.colsCountValue));

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

    DoubleMatrix& MultiplyByValue(const double value)
    {
        for (int i = 0; i < rowsCountValue; i++)
        {
            for (int j = 0; j < colsCountValue; j++)
            {
                operator()(i, j) *= value;
            }
        }
        return *this;
    }

    friend std::shared_ptr<DoubleMatrix> operator* (const double b, const DoubleMatrix & a)
    {
        auto res = std::shared_ptr<DoubleMatrix>(new DoubleMatrix(a));
        res->MultiplyByValue(b);
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

    std::shared_ptr<DoubleMatrix> CropMatrix(int startRow, int rowsCount) const
    {
        auto res = std::shared_ptr<DoubleMatrix>(new DoubleMatrix(rowsCount, colsCountValue));
        for (int i = startRow; i < startRow + rowsCount; i++)
        {
            int iIndex = i - startRow;
            for (int j = 0; j < colsCountValue; j++)
            {
                (*res)(iIndex, j) = operator ()(i, j);
            }
        }
        return res;
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
