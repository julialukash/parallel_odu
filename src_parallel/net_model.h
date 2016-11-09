#ifndef NETMODEL_H
#define NETMODEL_H

#include "interface.h"
#include <vector>

//#define DEBUG

class NetModel
{
private:
    const double q = 2.0/3.0;
    double xStepValue, yStepValue;
    double xAverageStepValue, yAverageStepValue;
    double xStartStepValue, yStartStepValue;
public:
    std::vector<double> xValues, yValues;
    double xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary;
    int xPointsCount, yPointsCount;

    double xValue(int i) const
    {
        auto tmp = xValues[i];
        auto eq = fabs(tmp - xMinBoundary - i * xStepValue) < 1e-4;
#ifdef DEBUG
//        std::cout << "i = " << i << ", xV[i] = " << xValues[i] << ", old = " << xMinBoundary + i * xStepValue << ", eq = " << eq << std::endl;
#endif
 //        return xMinBoundary + i * xStepValue;
        return tmp;
    }

    double yValue(int i) const
    {        
        auto tmp = yValues[i];
        auto eq = fabs(tmp - yMinBoundary - i * yStepValue) < 1e-4;
#ifdef DEBUG
//        std::cout << "j = " << i << ", yV[i] = " << yValues[i] << ", old = " <<
//                     yMinBoundary + i * yStepValue << ", eq = " << eq << std::endl;
#endif
        //        return yMinBoundary + i * yStepValue;
        return tmp;
    }

    inline double xStep(int i) const
    {
        auto tmp = xValues[i + 1] - xValues[i];
        return tmp;
    }

    inline double yStep(int i) const
    {
        auto tmp = yValues[i + 1] - yValues[i];
        auto isEq = fabs(tmp - yStepValue) < 1e-4;
#ifdef DEBUG
        std::cout << "i = " << i << ", yV + 1 = " << yValues[i + 1] << ", yValues[i] = "
                  << yValues[i] << ", tmp = " << tmp << ", old" << yStepValue
                  << ", isEq = " << isEq
                  << std::endl;
#endif
        return tmp;
    }

    inline double xAverageStep(int i) const
    {
        auto tmp = 0.5 * (xStep(i) + xStep(i - 1));
        return tmp;
    }

    inline double yAverageStep(int i) const
    {
        auto tmp = 0.5 * (yStep(i) + yStep(i - 1));
        auto isEq = fabs(tmp - yAverageStepValue) < 1e-4;
#ifdef DEBUG
        std::cout << "i = " << i << ", yS = " << yStep(i) << ", yS - 1 = "
                  << yStep(i - 1) << ", tmp = " << tmp << ", old = " << yAverageStepValue
                  << ", isAEq = " << isEq
                  << std::endl;
#endif
        return tmp;
    }

    NetModel()
    {
    }

    ~NetModel()
    {
        xValues.clear();
        yValues.clear();
    }

    NetModel(double xMinBoundaryValue, double xMaxBoundaryValue, double yMinBoundaryValue, double yMaxBoundaryValue,
             int xPointsCountValue, int yPointsCountValue)
    {
        xMinBoundary = xMinBoundaryValue;
        xMaxBoundary = xMaxBoundaryValue;
        yMinBoundary = yMinBoundaryValue;
        yMaxBoundary = yMaxBoundaryValue;
        xPointsCount = xPointsCountValue;
        yPointsCount = yPointsCountValue;
        xStepValue = (xMaxBoundary - xMinBoundary) / (xPointsCountValue - 1);
        yStepValue = (yMaxBoundary - yMinBoundary) / (yPointsCountValue - 1);
        xAverageStepValue = xStepValue;
        yAverageStepValue = yStepValue;        
    }

    void InitModelNorm(int firstRowIndex, int lastRowIndex, int firstColIndex, int lastColIndex)
    {
        for (int i = firstColIndex - 1; i <= lastColIndex + 1; ++i)
        {
            xValues.push_back(xMinBoundary + i * xStepValue);
        }

        for (int i = firstRowIndex - 1; i <= lastRowIndex + 1; ++i)
        {
            yValues.push_back(yMinBoundary + i * yStepValue);
        }
    }

    double f(double x)
    {
        return (pow(1.0 + x, q) - 1.0) / (pow(2.0, q) - 1.0);
    }

    void InitModel(int firstRowIndex, int lastRowIndex, int firstColIndex, int lastColIndex)
    {
        for (int i = firstColIndex; i <= lastColIndex + 1; ++i)
        {
            xValues.push_back(xMaxBoundary * f(1.0 * i / (xPointsCount - 1)));
        }

        for (int i = firstRowIndex - 1; i <= lastRowIndex + 1; ++i)
        {
            yValues.push_back(yMaxBoundary * f(1.0 * i / (yPointsCount - 1)));
        }
    }


    bool IsInnerPoint(int j) const
    {
        return j == 0 || j == yPointsCount - 1;
    }
};

#endif // NETMODEL_H
