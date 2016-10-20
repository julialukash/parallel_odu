#ifndef NETMODEL_H
#define NETMODEL_H

#include "interface.h"

class NetModel
{
private:
    const double q=2.0/3.0;
    double xStepValue, yStepValue;
    double xAverageStepValue, yAverageStepValue;
    std::vector<double> xStepValues, yStepValues;
    std::vector<double> xValues, yValues;
public:
    double xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary;
    double xPointsCount, yPointsCount;

    inline double xValue(int i) { return xValues[i]; }
    inline double yValue(int i) { return yValues[i]; }

    inline double xStep(int i) { return xStepValues[i]; }
    inline double yStep(int i) { return yStepValues[i]; }

    inline double xAverageStep(int i) { return 0.5 * (xStepValues[i] + xStepValues[i - 1]); }
    inline double yAverageStep(int i) { return 0.5 * (yStepValues[i] + yStepValues[i - 1]); }

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
        auto xStepValueInit = (xMaxBoundary - xMinBoundary) / xPointsCount;
        auto yStepValueInit = (yMaxBoundary - yMinBoundary) / yPointsCount;
        xStepValues.push_back(xStepValueInit);
        xValues.push_back(xMinBoundary);
        yStepValues.push_back(yStepValueInit);
        for (auto i = 1; i < xPointsCount; ++i) {
            xStepValues.push_back(q * xStepValues[i - 1]);
            xValues.push_back(xValues[i - 1] + xStepValues[i]);
        }
        xValues.push_back(xMaxBoundary);
        for (auto i = 1; i < yPointsCount; ++i) {
            yStepValues.push_back(q * yStepValues[i - 1]);
        }
        xPointsCount = xPointsCount;
        yPointsCount = yPointsCount;
    }
};

#endif // NETMODEL_H
