#ifndef NETMODEL_H
#define NETMODEL_H

#include "interface.h"

class NetModel
{
private:
    const double coefficient = 2.0/3.0;
    double xStepValue, yStepValue;
    double xAverageStepValue, yAverageStepValue;
    double xStartStepValue, yStartStepValue;
public:
    double xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary;
    int xPointsCount, yPointsCount;

    double xValue(int i) const
    {
        return xMinBoundary + i * xStepValue;
    }

    double yValue(int i) const
    {
        return yMinBoundary + i * yStepValue;
    }

    inline double xStep(int i) const { return xStepValue; }
    inline double yStep(int i) const { return yStepValue; }

    inline double xAverageStep(int i) const { return xAverageStepValue; }
    inline double yAverageStep(int i) const { return yAverageStepValue; }

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
        double denominator = 0.0;
        for (int i = 0; i < xPointsCountValue; ++i)
        {
            denominator += pow(coefficient, i);
        }
        xStartStepValue = (xMaxBoundaryValue - xMinBoundaryValue) / denominator;

        denominator = 0;
        for (int i = 0; i < yPointsCountValue; ++i)
        {
            denominator += pow(coefficient, i);
        }
        yStartStepValue = (yMaxBoundaryValue - yMinBoundaryValue) / denominator;
    }

    bool IsInnerPoint(int i, int j) const
    {
        return i == 0 || i == xPointsCount - 1 || j == 0 || j == yPointsCount - 1;
    }
};

#endif // NETMODEL_H
