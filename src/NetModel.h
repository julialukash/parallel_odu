#ifndef NETMODEL_H
#define NETMODEL_H

#include "interface.h"

class NetModel
{
public:
    double_matrix values;
    double xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary;
    double xStep, yStep;
    double xAverageStep, yAverageStep;

    inline size_t xSize() { return values.size1(); }
    inline size_t ySize() { return values.size2(); }

    inline double xValue(int i) { return i * xStep; }
    inline double yValue(int i) { return i * yStep; }


    NetModel(double xMinBoundary, double xMaxBoundary, double yMinBoundary, double yMaxBoundary,
             int xPointsCount, int yPointsCount)
    {
        xStep = (xMaxBoundary - xMinBoundary) / xPointsCount;
        yStep = (yMaxBoundary - yMinBoundary) / yPointsCount;
        xAverageStep = xStep;
        yAverageStep = yStep;
        values = double_matrix(xPointsCount + 1, yPointsCount + 1, 0);
    }

    NetModel(NetModel otherModel)
    {
        xStep = otherModel.xStep;
        yStep = otherModel.yStep;
        xAverageStep = otherModel.xAverageStep;
        yAverageStep = otherModel.yAverageStep;
        values = double_matrix(otherModel.values.size1(), otherModel.values.size2(), 0);
    }
};

#endif // NETMODEL_H
