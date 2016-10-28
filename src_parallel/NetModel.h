#ifndef NETMODEL_H
#define NETMODEL_H

#include "Interface.h"

class NetModel
{
private:
    double xStepValue, yStepValue;
    double xAverageStepValue, yAverageStepValue;
public:
    double xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary;
    int xPointsCount, yPointsCount;

    inline double xValue(int i) { return xMinBoundary + i * xStepValue; }
    inline double yValue(int i) { return yMinBoundary + i * yStepValue; }

    inline double xStep(int i) { return xStepValue; }
    inline double yStep(int i) { return yStepValue; }

    inline double xAverageStep(int i) { return xAverageStepValue; }
    inline double yAverageStep(int i) { return yAverageStepValue; }

    NetModel()
    {

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

    bool IsInnerPoint(int i, int j)
    {
        return i == 0 || i == xPointsCount - 1 || j == 0 || j == yPointsCount - 1;
    }

};

#endif // NETMODEL_H
