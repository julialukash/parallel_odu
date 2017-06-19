#ifndef CONJUGATEGRADIENTALGO_H
#define CONJUGATEGRADIENTALGO_H

#include "approximate_operations.h"
#include "differential_equation_model.h"
#include "interface.h"
#include "processors_data.h"

class ConjugateGradientAlgo
{
private:
    const ProcessorsData& processorData;
    const NetModel& netModel;
    const DifferentialEquationModel& diffModel;
    const ApproximateOperations& approximateOperations;

    double CalculateTauValue(const DoubleMatrix& residuals, const DoubleMatrix& grad, const DoubleMatrix& laplassGrad);
    double CalculateAlphaValue(const DoubleMatrix& laplassResiduals, const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad);
    DoubleMatrix* CalculateResidual(const DoubleMatrix &p);
    DoubleMatrix* CalculateGradient(const DoubleMatrix& residuals, const DoubleMatrix& laplassResiduals,
                                   const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad,
                                   int k);
    DoubleMatrix* CalculateNewP(const DoubleMatrix &p, const DoubleMatrix &grad, double tau);
    double CalculateError(const DoubleMatrix &uValues, const DoubleMatrix &p);
    bool IsStopCondition(const DoubleMatrix &p, const DoubleMatrix &previousP);
    DoubleMatrix* Init();
    void RenewBounds(DoubleMatrix &values);
public:
    ConjugateGradientAlgo(const NetModel& model, const DifferentialEquationModel& modelDiff,
                          const ApproximateOperations& approximateOperationsPtr,
                          const ProcessorsData& processorDataPtr);
    DoubleMatrix* CalculateU();
    DoubleMatrix* Process(double* error, const DoubleMatrix &uValues);
};

#endif // CONJUGATEGRADIENTALGO_H
