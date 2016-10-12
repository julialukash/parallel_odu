#ifndef DERIVATOR_H
#define DERIVATOR_H

#include "interface.h"
#include "NetModel.h"

class Derivator
{
public:
    Derivator();

    NetModel CalculateLaplassApproximately(NetModel currentValues)
    {
        auto laplassValues = NetModel(currentValues);
        for (auto i = 1; i < laplassValues.xSize() - 1; ++i)
        {
            for (auto j = 1; j < laplassValues.ySize() - 1; ++j)
            {
                laplassValues[i, j] = - (2 * currentValues[i, j] - currentValues[i - 1, j] - currentValues[i + 1, j]) /
                                        (laplassValues.xAverageStep * laplassValues.xStep) +
                                        (2 * currentValues[i, j] - currentValues[i, j - 1] - currentValues[i, j + 1])  /
                                        (laplassValues.yAverageStep * laplassValues.yStep);
            }
        }
        return laplassValues;
    }

    double ScalarProduct(NetModel currentValues, NetModel otherValues)
    {
        double prodValue = 0;
        for (auto i = 1; i < currentValues.xSize() - 1; ++i)
        {
            for (auto j = 1; j < currentValues.ySize() - 1; ++j)
            {
                prodValue = prodValue + currentValues.xAverageStep * currentValues.yAverageStep *
                                        currentValues[i, j] * otherValues[i, j];
            }
        }
        return prodValue;
    }

    double NormValue(NetModel currentValues)
    {
        return sqrt(ScalarProduct(currentValues, currentValues));
    }
};

#endif // DERIVATOR_H
