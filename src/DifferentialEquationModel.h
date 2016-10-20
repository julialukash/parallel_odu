#ifndef MODEL_H
#define MODEL_H

class DifferentialEquationModel
{
private:
public:
    DifferentialEquationModel()
    {

    }

    double CalculateFunctionValue(double x, double y)
    {
        return (x * x + y * y) * sin(x * y);
    }

    double CalculateBoundaryValue(double x, double y)
    {
          return 1 + sin(x * y);
    }
};

#endif // MODEL_H
