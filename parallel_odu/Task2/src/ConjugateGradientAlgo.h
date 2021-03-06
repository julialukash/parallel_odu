#ifndef CONJUGATEGRADIENTALGO_H
#define CONJUGATEGRADIENTALGO_H

#include "ApproximateOperations.h"
#include "DifferentialEquationModel.h"
#include "Interface.h"

class ConjugateGradientAlgo
{
private:
    const double eps = 10e-4;

    std::shared_ptr<NetModel> netModel;
    std::shared_ptr<DifferentialEquationModel> diffModel;
    std::shared_ptr<ApproximateOperations> approximateOperations;

    double CalculateTauValue(const DoubleMatrix& residuals, const DoubleMatrix& grad, const DoubleMatrix& laplassGrad);
    double CalculateAlphaValue(const DoubleMatrix& laplassResiduals, const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad);
    DoubleMatrix CalculateResidual(const DoubleMatrix &p);
    DoubleMatrix CalculateGradient(const DoubleMatrix& residuals, const DoubleMatrix& laplassResiduals,
                                   const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad,
                                   int k);
    DoubleMatrix CalculateNewP(const DoubleMatrix &p, const DoubleMatrix &grad, double tau);
    double CalculateError(const DoubleMatrix &uValues, const DoubleMatrix &p);
    bool IsStopCondition(const DoubleMatrix &p, const DoubleMatrix &previousP);

public:
    ConjugateGradientAlgo(std::shared_ptr<NetModel> model, std::shared_ptr<DifferentialEquationModel> modelDiff,
                      std::shared_ptr<ApproximateOperations> approximateOperationsPtr);
    DoubleMatrix Init();
    double Process(DoubleMatrix &initP, const DoubleMatrix &uValues);
};

#endif // CONJUGATEGRADIENTALGO_H
