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
    double xPointsCount, yPointsCount;

    inline double xValue(int i) { return i * xStepValue; }
    inline double yValue(int i) { return i * yStepValue; }

    inline double xStep(int i) { return xStepValue; }
    inline double yStep(int i) { return yStepValue; }

    inline double xAverageStep(int i) { return xAverageStepValue; }
    inline double yAverageStep(int i) { return yAverageStepValue; }

    NetModel()
    {

    }

    NetModel(double xMinBoundaryValue, double xMaxBoundaryValue, double yMinBoundaryValue, double yMaxBoundaryValue,
             long xPointsCount, long yPointsCount)
    {
        xMinBoundary = xMinBoundaryValue;
        xMaxBoundary = xMaxBoundaryValue;
        yMinBoundary = yMinBoundaryValue;
        yMaxBoundary = yMaxBoundaryValue;
        xStepValue = (xMaxBoundary - xMinBoundary) / xPointsCount;
        yStepValue = (yMaxBoundary - yMinBoundary) / yPointsCount;
        xAverageStepValue = xStepValue;
        yAverageStepValue = yStepValue;
        xPointsCount = xPointsCount;
        yPointsCount = yPointsCount;
    }
};

#endif // NETMODEL_H
