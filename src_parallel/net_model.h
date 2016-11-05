#ifndef NETMODEL_H
#define NETMODEL_H

#include "interface.h"
#include <vector>

class NetModel
{
private:
    const double coefficient = 2.0/3.0;
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
        auto eq = fabs(tmp - xMinBoundary + i * xStepValue) < 1e-4;
        std::cout << "i = " << i << ", xV[i] = " << xValues[i] << ", old = " << xMinBoundary + i * xStepValue << ", eq = " << eq << std::endl;
//        return xMinBoundary + i * xStepValue;
        return tmp;
    }

    double yValue(int i) const
    {        
        auto tmp = yValues[i];
        auto eq = fabs(tmp - yMinBoundary + i * yStepValue) < 1e-4;
        std::cout << "j = " << i << ", yV[i] = " << yValues[i] << ", old = " << yMinBoundary + i * yStepValue << ", eq = " << eq << std::endl;
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
        std::cout << "i = " << i << ", yV + 1 = " << yValues[i + 1] << ", yValues[i] = "
                  << yValues[i] << ", tmp = " << tmp << ", old" << yStepValue
                  << ", isEq = " << isEq
                  << std::endl;
        return tmp;
    }

    inline double xAverageStep(int i) const
    {
        auto tmp = 0.5 * (xValues[i + 1] - xValues[i - 1]);
        return xAverageStepValue;
    }

    inline double yAverageStep(int i) const
    {
        auto tmp = 0.5 * (yValues[i + 1] - yValues[i - 1]);
        return yAverageStepValue;
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
        xPointsCount = xPointsCountValue + 1;
        yPointsCount = yPointsCountValue + 1;
        xStepValue = (xMaxBoundary - xMinBoundary) / xPointsCountValue;
        yStepValue = (yMaxBoundary - yMinBoundary) / yPointsCountValue;
        xAverageStepValue = xStepValue;
        yAverageStepValue = yStepValue;        
    }

    void InitModel(int firstRowIndex, int lastRowIndex, int firstColIndex, int lastColIndex)
    {
        for (int i = firstColIndex; i <= lastColIndex; ++i)
        {
            xValues.push_back(xMinBoundary + i * xStepValue);
        }

        for (int i = firstRowIndex - 1; i <= lastRowIndex + 1; ++i)
        {
            yValues.push_back(yMinBoundary + i * yStepValue);
        }
    }

    bool IsInnerPoint(int j) const
    {
        return j == 0 || j == yPointsCount - 1;
    }
};

#endif // NETMODEL_H
