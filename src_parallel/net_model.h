#ifndef NETMODEL_H
#define NETMODEL_H

#include "interface.h"
#include <vector>

//#define DEBUG

class NetModel
{
private:
    const double q = 2.0/3.0;
public:
    std::vector<double> xValues, yValues;
    double xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary;
    int xPointsCount, yPointsCount;

    double xValue(int i) const
    {
        return xValues[i];
    }

    double yValue(int i) const
    {
        return yValues[i];
    }

    inline double xStep(int i) const
    {
        return xValues[i + 1] - xValues[i];
    }

    inline double yStep(int i) const
    {
        return yValues[i + 1] - yValues[i];
    }

    inline double xAverageStep(int i) const
    {
        return 0.5 * (xStep(i) + xStep(i - 1));
    }

    inline double yAverageStep(int i) const
    {
        return 0.5 * (yStep(i) + yStep(i - 1));
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
    }

    double f(double x)
    {
        return (pow(1.0 + x, q) - 1.0) / (pow(2.0, q) - 1.0);
    }

    void InitModel(int firstRowIndex, int lastRowIndex, int firstColIndex, int lastColIndex)
    {
        for (int i = firstColIndex - 1; i <= lastColIndex + 2; ++i)
        {
            xValues.push_back(xMaxBoundary * f(1.0 * i / (xPointsCount - 1)));
        }

        for (int i = firstRowIndex - 1; i <= lastRowIndex + 2; ++i)
        {
            yValues.push_back(yMaxBoundary * f(1.0 * i / (yPointsCount - 1)));
        }
    }
};

#endif // NETMODEL_H
