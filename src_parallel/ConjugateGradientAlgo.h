#ifndef CONJUGATEGRADIENTALGO_H
#define CONJUGATEGRADIENTALGO_H

#include "ApproximateOperations.h"
#include "DifferentialEquationModel.h"
#include "Interface.h"
#include "ProcessorsData.h"

class ConjugateGradientAlgo
{
private:
    const double eps = 10e-4;
    const ProcessorsData& processorData;
    const NetModel& netModel;
    const DifferentialEquationModel& diffModel;
    const ApproximateOperations& approximateOperations;

    double CalculateTauValue(const DoubleMatrix& residuals, const DoubleMatrix& grad, const DoubleMatrix& laplassGrad);
    double CalculateAlphaValue(const DoubleMatrix& laplassResiduals, const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad);
    std::shared_ptr<DoubleMatrix> CalculateResidual(const DoubleMatrix &p);
    std::shared_ptr<DoubleMatrix> CalculateGradient(std::shared_ptr<DoubleMatrix> residuals, const DoubleMatrix& laplassResiduals,
                                   const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad,
                                   int k);
    std::shared_ptr<DoubleMatrix> CalculateNewP(const DoubleMatrix &p, const DoubleMatrix &grad, double tau);
    double CalculateError(const DoubleMatrix &uValues, const DoubleMatrix &p);
    bool IsStopCondition(const DoubleMatrix &p, const DoubleMatrix &previousP);
public:
    ConjugateGradientAlgo(const NetModel& model, const DifferentialEquationModel& modelDiff,
                          const ApproximateOperations& approximateOperationsPtr,
                          const ProcessorsData& processorDataPtr);
    std::shared_ptr<DoubleMatrix> Init();
    std::shared_ptr<DoubleMatrix> CalculateU();
    double Process(std::shared_ptr<DoubleMatrix>& initP, const DoubleMatrix &uValues);
    void RenewBoundRows(DoubleMatrix &values);
};

#endif // CONJUGATEGRADIENTALGO_H
