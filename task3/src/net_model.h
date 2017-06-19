#ifndef NETMODEL_H
#define NETMODEL_H

#include <iostream>

#include "interface.h"

#include "cuda_runtime.h"


class NetModel
{
private:
public:
    int xSize, ySize;
    double q;
    double xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary;
    int firstRowIndex, firstColIndex;
    int lastRowIndex, lastColIndex;
    int xPointsCount, yPointsCount;


    __device__ double xValue(double* devXValues, const int i) const
    {
        return devXValues[i];
    }

    __device__ double yValue(double* devYValues, const int i) const
    {
        return devYValues[i];
    }


    __device__ double xStep(double* devXValues, const int i) const
    {
        return xValue(devXValues, i + 1) - xValue(devXValues, i);
    }

    __device__ double yStep(double* devYValues, const int i) const
    {
        return yValue(devYValues, i + 1) - yValue(devYValues, i);
    }

    __device__ double xAverageStep(double* devXValues, int i) const
    {
        return 0.5 * (xStep(devXValues, i) + xStep(devXValues, i - 1));
    }

    __device__ double yAverageStep(double* devYValues, int i) const
    {
        return 0.5 * (yStep(devYValues, i) + yStep(devYValues, i - 1));
    }

    NetModel() { }

    ~NetModel(){ }

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

    double f(double x, double qValue) const
    {
        return (pow(1.0 + x, qValue) - 1.0) / (pow(2.0, qValue) - 1.0);
    }

    void InitModel(int firstRowIndexValue, int lastRowIndexValue, int firstColIndexValue, int lastColIndexValue)
    {
        q = 2.0 / 3.0;
        firstRowIndex = firstRowIndexValue - 1;
        firstColIndex = firstColIndexValue - 1;
        lastRowIndex = lastRowIndexValue;
        lastColIndex = lastColIndexValue;
        xSize = lastColIndex + 2 - (firstColIndexValue - 1) + 1;
        ySize = lastRowIndex + 2 - (firstRowIndexValue - 1) + 1;
    }
};

#endif // NETMODEL_H
