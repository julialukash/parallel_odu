#ifndef MODEL_H
#define MODEL_H

class DifferentialEquationModel
{
private:
public:
    double CalculateFunctionValue(double x, double y)
    {
        return 1 + x * y == 0 ? 0 : ((x * x + y * y) / (1 + x * y)^2);
    }

    double CalculateBoundaryValue(double x, double y)
    {
        return 1 + x * y > 0 ? 0 : ln(1 + x * y);
    }
};

#endif // MODEL_H
