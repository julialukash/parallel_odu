#ifndef MODEL_H
#define MODEL_H

#include "interface.h"

class DifferentialEquationModel
{
private:
public:
    DifferentialEquationModel() {}

    double CalculateFunctionValue(double x, double y) const
    {
        return (x * x + y * y) * sin(x * y);
    }

    double CalculateBoundaryValue(double x, double y) const
    {
          return 1 + sin(x * y);
    }

    double CalculateUValue(double x, double y) const
    {
        return 1 + sin(x * y);
    } 
};

#endif // MODEL_H
