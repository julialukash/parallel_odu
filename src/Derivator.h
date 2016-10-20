#ifndef DERIVATOR_H
#define DERIVATOR_H

#include "interface.h"

class Derivator
{
private:
    std::shared_ptr<NetModel> netModel;
public:
    Derivator(std::shared_ptr<NetModel> model)
    {
        netModel = model;
    }

    // note: calculates -laplass(currentValues)
    double_matrix CalculateLaplassApproximately(double_matrix currentValues)
    {
        auto laplassValues = double_matrix(netModel->xPointsCount, netModel->yPointsCount);
        for (auto i = 1; i < laplassValues.size1() - 1; ++i)
        {
            for (auto j = 1; j < laplassValues.size2() - 1; ++j)
            {
                auto xPart = (currentValues(i, j) - currentValues(i - 1, j)) / netModel->xStep(i - 1) -
                             (currentValues(i + 1, j) - currentValues(i, j)) / netModel->xStep(i);
                auto yPart = (currentValues(i, j) - currentValues(i, j - 1)) / netModel->yStep(i - 1) -
                             (currentValues(i, j + 1) - currentValues(i, j)) / netModel->yStep(i);
                laplassValues(i, j) = xPart / netModel->xAverageStep(i) + yPart / netModel->yAverageStep(j);
            }
        }
        return laplassValues;
    }

    double ScalarProduct(double_matrix currentValues, double_matrix otherValues)
    {
        double prodValue = 0;
        for (auto i = 1; i < currentValues.size1() - 1; ++i)
        {
            for (auto j = 1; j < currentValues.size2() - 1; ++j)
            {
                prodValue = prodValue + netModel->xAverageStep(i) * netModel->yAverageStep(j) *
                                        currentValues(i, j) * otherValues(i, j);
            }
        }
        return prodValue;
    }

    double NormValue(double_matrix currentValues)
    {
        return sqrt(ScalarProduct(currentValues, currentValues));
    }
};

#endif // DERIVATOR_H
