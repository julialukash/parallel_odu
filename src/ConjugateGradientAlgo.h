#ifndef CONJUGATEGRADIENTALGO_H
#define CONJUGATEGRADIENTALGO_H

#include "Interface.h"
#include "ApproximateOperations.h"
#include "DifferentialEquationModel.h"

class ConjugateGradientAlgo
{
private:
    const double eps = 10e-4;

    std::shared_ptr<NetModel> netModel;
    std::shared_ptr<DifferentialEquationModel> diffModel;
    std::shared_ptr<ApproximateOperations> approximateOperations;

    double CalculateTauValue(double_matrix residuals, double_matrix grad, double_matrix laplassGrad);
    double CalculateAlphaValue(double_matrix laplassResiduals, double_matrix previousGrad, double_matrix laplassPreviousGrad);
    double_matrix CalculateResidual(double_matrix p);
    double_matrix CalculateGradient(double_matrix residuals, double_matrix laplassResiduals,
                                    double_matrix previousGrad, double_matrix laplassPreviousGrad,
                                    int k);
    double_matrix CalculateNewP(double_matrix p, double_matrix grad, double tau);
    double CalculateError(double_matrix uValues, double_matrix p);
    bool IsStopCondition(double_matrix p, double_matrix previousP);

public:
    ConjugateGradientAlgo(std::shared_ptr<NetModel> model, std::shared_ptr<DifferentialEquationModel> modelDiff,
                      std::shared_ptr<ApproximateOperations> approximateOperationsPtr);
    double_matrix Init();
    double_matrix Process(double_matrix initP, double_matrix uValues);
};

#endif // CONJUGATEGRADIENTALGO_H
