#include <iostream>

#include "conjugate_gradient_algo.h"
#include "mpi_operations.h"


ConjugateGradientAlgo::ConjugateGradientAlgo(const NetModel& modelNet, const DifferentialEquationModel& modelDiff,
                                             const ApproximateOperations& approximateOperations,
                                             const ProcessorsData& processorData):
    processorData(processorData), netModel(modelNet),
    diffModel(modelDiff),  approximateOperations(approximateOperations)
{
}

std::shared_ptr<DoubleMatrix> ConjugateGradientAlgo::CalculateU()
{
    auto values = std::shared_ptr<DoubleMatrix>(new DoubleMatrix(processorData.RowsCount(), netModel.yPointsCount));
    for (int i = 0; i < values->rowsCount(); ++i)
    {
        int iNetIndex = i + processorData.FirstRowIndex();
        for (int j = 0; j < values->colsCount(); ++j)
        {
            (*values)(i, j) = diffModel.CalculateUValue(netModel.xValue(iNetIndex), netModel.yValue(j));
        }
    }
    return values;
}

std::shared_ptr<DoubleMatrix> ConjugateGradientAlgo::Init()
{
    auto values = std::shared_ptr<DoubleMatrix>(new DoubleMatrix(processorData.RowsCountWithBorders(), netModel.xPointsCount));
    for (int i = 0; i < values->rowsCount(); ++i)
    {
        auto iValueIndex = processorData.IsFirstProcessor() ? i + 1 : i;
        auto iNetIndex = i + processorData.FirstRowWithBordersIndex();
        if (iNetIndex  >= processorData.FirstRowIndex() && iNetIndex <= processorData.LastRowIndex())
        {
            for (int j = 0; j < values->colsCount(); ++j)
            {
                if (netModel.IsInnerPoint(iNetIndex, j))
                {
                    (*values)(iValueIndex, j) = diffModel.CalculateBoundaryValue(netModel.xValue(iNetIndex), netModel.yValue(j));
                }
                else
                {
                    (*values)(iValueIndex, j) = 0.0;
                }
            }
        }
    }
    return values;
}

void ConjugateGradientAlgo::RenewBoundRows(DoubleMatrix& values)
{
    RenewMatrixBoundRows(values, processorData, netModel);
}

double ConjugateGradientAlgo::Process(std::shared_ptr<DoubleMatrix>& p, const DoubleMatrix& uValues)
{
    std::shared_ptr<DoubleMatrix> previousP, grad, laplassGrad, laplassPreviousGrad;
    previousP = p; laplassGrad = p; laplassPreviousGrad = p; grad = p;
    int iteration = 0;
    double error = -1.0;
    while (true)
    {
        RenewBoundRows(*p);
        error = CalculateError(uValues, *p);

        // check stop condition
        auto stopCondition = iteration != 0 && IsStopCondition(*p, *previousP);
        if (stopCondition)
        {
            auto pCroppedPtr = p->CropMatrix(1, p->rowsCount() - 2);
            p.reset();
            p = pCroppedPtr;
            break;
        }
        laplassPreviousGrad.swap(laplassGrad);

        auto residuals = CalculateResidual(*p);
        RenewBoundRows(*residuals);

        auto laplassResiduals = approximateOperations.CalculateLaplass(*residuals, processorData);
        RenewBoundRows(*laplassResiduals);

        grad = CalculateGradient(residuals, *laplassResiduals, *grad, *laplassPreviousGrad, iteration);
        laplassGrad = approximateOperations.CalculateLaplass(*grad, processorData);
        RenewBoundRows(*laplassGrad);

        auto tau = CalculateTauValue(*residuals, *grad, *laplassGrad);
        previousP.swap(p);
        p = CalculateNewP(*previousP, *grad, tau);

        ++iteration;        
    }
    return error;
}

double ConjugateGradientAlgo::CalculateTauValue(const DoubleMatrix& residuals, const DoubleMatrix& grad, const DoubleMatrix& laplassGrad)
{    
    double numerator = approximateOperations.ScalarProduct(residuals, grad, processorData);
    double denominator = approximateOperations.ScalarProduct(laplassGrad, grad, processorData);
    double tauValue = GetFractionValueFromAllProcessors(numerator, denominator);
    return tauValue;
}

double ConjugateGradientAlgo::CalculateAlphaValue(const DoubleMatrix& laplassResiduals, const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad)
{
    double numerator = approximateOperations.ScalarProduct(laplassResiduals, previousGrad, processorData);
    double denominator = approximateOperations.ScalarProduct(laplassPreviousGrad, previousGrad, processorData);
    double alphaValue = GetFractionValueFromAllProcessors(numerator, denominator);
    return alphaValue;
}

std::shared_ptr<DoubleMatrix> ConjugateGradientAlgo::CalculateResidual(const DoubleMatrix& p)
{
    auto laplassP = approximateOperations.CalculateLaplass(p, processorData);
    auto residuals = std::make_shared<DoubleMatrix>(laplassP->rowsCount(), laplassP->colsCount());
    int startIndex = 1;
    int endIndex = processorData.RowsCountWithBorders() - 2;
    for (int i = startIndex; i <= endIndex; ++i)
    {
        int iNetIndex = i - startIndex + processorData.FirstRowIndex();
        for (int j = 0; j < residuals->colsCount(); ++j)
        {
            if (netModel.IsInnerPoint(iNetIndex, j))
            {
                (*residuals)(i, j) = 0;
            }
            else
            {
                (*residuals)(i, j) = (*laplassP)(i, j) - diffModel.CalculateFunctionValue(netModel.xValue(j), netModel.yValue(iNetIndex));
            }
        }
    }
    return residuals;
}

std::shared_ptr<DoubleMatrix> ConjugateGradientAlgo::CalculateGradient(std::shared_ptr<DoubleMatrix> residuals, const DoubleMatrix& laplassResiduals,
                                const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad,
                                int k)
{
    std::shared_ptr<DoubleMatrix> gradient;
    if (k == 0)
    {
        gradient = residuals;
    }
    else
    {
        double alpha = CalculateAlphaValue(laplassResiduals, previousGrad, laplassPreviousGrad);
        gradient = *residuals - *(alpha * previousGrad);
    }
    return gradient;
}

std::shared_ptr<DoubleMatrix> ConjugateGradientAlgo::CalculateNewP(const DoubleMatrix& p, const DoubleMatrix& grad, double tau)
{
    return p - *(tau * grad);
}

double ConjugateGradientAlgo::CalculateError(const DoubleMatrix& uValues, const DoubleMatrix& p)
{
    auto pCropped = p.CropMatrix(1, p.rowsCount() - 2);
    auto psi = uValues - *pCropped;
    double error = approximateOperations.MaxNormValue(*psi);
    double globalError = GetMaxValueFromAllProcessors(error);
    return globalError;
}

bool ConjugateGradientAlgo::IsStopCondition(const DoubleMatrix& p, const DoubleMatrix& previousP)
{
    auto pDiff = p - previousP;
    auto pDiffCropped = pDiff->CropMatrix(1, pDiff->rowsCount() - 2);
    double pDiffNormLocal, pDiffNormGlobal;
    pDiffNormLocal = approximateOperations.MaxNormValue(*pDiffCropped);
    pDiffNormGlobal = GetMaxValueFromAllProcessors(pDiffNormLocal);
    return pDiffNormGlobal < eps;
}
