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

    double_matrix CalculateUValues(std::shared_ptr<NetModel> netModel)
    {
        auto values = double_matrix(netModel->xPointsCount, netModel->yPointsCount);
        for (size_t i = 0; i < values.size1(); ++i)
        {
            for (size_t j = 0; j < values.size2(); ++j)
            {
                values(i, j) = CalculateUValue(netModel->xValue(i), netModel->yValue(j));
            }
        }
        return values;
    }
};

#endif // MODEL_H
