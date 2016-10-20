#ifndef NETMODEL_H
#define NETMODEL_H

#include "interface.h"

class NetModel
{
private:
    double xStepValue, yStepValue;
    double xAverageStepValue, yAverageStepValue;
public:
    double xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary;
    long xPointsCount, yPointsCount;

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
             long xPointsCountValue, long yPointsCountValue)
    {
        xMinBoundary = xMinBoundaryValue;
        xMaxBoundary = xMaxBoundaryValue;
        yMinBoundary = yMinBoundaryValue;
        yMaxBoundary = yMaxBoundaryValue;
        xPointsCount = xPointsCountValue;
        yPointsCount = yPointsCountValue;
        xStepValue = (xMaxBoundary - xMinBoundary) / xPointsCount;
        yStepValue = (yMaxBoundary - yMinBoundary) / yPointsCount;
        xAverageStepValue = xStepValue;
        yAverageStepValue = yStepValue;
    }
};

#endif // NETMODEL_H
