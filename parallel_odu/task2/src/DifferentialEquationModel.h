#ifndef MODEL_H
#define MODEL_H

#include "Interface.h"

class DifferentialEquationModel
{
private:
public:
    DifferentialEquationModel() {}

    double CalculateFunctionValue(double x, double y)
    {
        return (x * x + y * y) * sin(x * y);
    }

    double CalculateBoundaryValue(double x, double y)
    {
          return 1 + sin(x * y);
    }

    double CalculateUValue(double x, double y)
    {
        return 1 + sin(x * y);
    }

    DoubleMatrix CalculateUValues(std::shared_ptr<NetModel> netModel)
    {
        auto values = DoubleMatrix(netModel->xPointsCount, netModel->yPointsCount);
        for (auto i = 0; i < values.size1(); ++i)
        {
            for (auto j = 0; j < values.size2(); ++j)
            {
                values(i, j) = CalculateUValue(netModel->xValue(i), netModel->yValue(j));
            }
        }
        return values;
    }
};

#endif // MODEL_H
