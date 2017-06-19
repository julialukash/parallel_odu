#include <iostream>

#include "conjugate_gradient_algo.h"
#include "mpi_operations.h"

#define PARALLEL = 1

const double eps = 1e-4;

ConjugateGradientAlgo::ConjugateGradientAlgo(const NetModel& modelNet, const DifferentialEquationModel& modelDiff,
                                             const ApproximateOperations& approximateOperations,
                                             const ProcessorsData& processorData):
    processorData(processorData), netModel(modelNet),
    diffModel(modelDiff),  approximateOperations(approximateOperations)
{
}

DoubleMatrix* ConjugateGradientAlgo::CalculateU()
{
    DoubleMatrix* values = new DoubleMatrix(processorData.RowsCount(), processorData.ColsCount());
#ifdef PARALLEL
    #pragma omp parallel for
#endif
    for (int i = 0; i < values->rowsCount(); ++i)
    {
        for (int j = 0; j < values->colsCount(); ++j)
        {
            // +1 as bound rows go first (0)
            (*values)(i, j) = diffModel.CalculateUValue(netModel.xValue(j + 1), netModel.yValue(i + 1));
        }
    }
    return values;
}

DoubleMatrix* ConjugateGradientAlgo::Init()
{
    DoubleMatrix* values = new DoubleMatrix(processorData.RowsCountWithBorders(), processorData.ColsCountWithBorders());
#ifdef PARALLEL
    #pragma omp parallel for
#endif
    for (int i = processorData.FirstOwnRowRelativeIndex(); i <= processorData.LastOwnRowRelativeIndex(); ++i)
    {
        for (int j = processorData.FirstOwnColRelativeIndex(); j <= processorData.LastOwnColRelativeIndex(); ++j)
        {
            bool isInnerPoint = processorData.IsInnerIndices(i, j);
            if (isInnerPoint)
            {
                (*values)(i, j) = diffModel.CalculateBoundaryValue(netModel.xValue(j), netModel.yValue(i));
            }
            else
            {
                (*values)(i, j) = 0.0;
            }
        }
    }
    return values;
}

void ConjugateGradientAlgo::RenewBounds(DoubleMatrix& values)
{
    RenewMatrixBoundRows(values, processorData);
    RenewMatrixBoundCols(values, processorData);
}

DoubleMatrix* ConjugateGradientAlgo::Process(double* error, const DoubleMatrix& uValues)
{

    DoubleMatrix* p = Init();
    DoubleMatrix* previousP = NULL;
    DoubleMatrix* grad = NULL;
    DoubleMatrix* laplassGrad = NULL;
    DoubleMatrix* laplassPreviousGrad = NULL;
    int iteration = 0;
    double errorValue = -1.0;
    while (true)
    {
        RenewBounds(*p);
        errorValue = CalculateError(uValues, *p);
        // check stop condition
        bool stopCondition = iteration != 0 && IsStopCondition(*p, *previousP);
        if (stopCondition)
        {
            DoubleMatrix* pCroppedPtr = p->CropMatrix(processorData.FirstOwnRowRelativeIndex(), processorData.RowsCount(),
                                             processorData.FirstOwnColRelativeIndex(), processorData.ColsCount());
            delete p;
            p = pCroppedPtr;
            break;
        }

        delete laplassPreviousGrad;
        laplassPreviousGrad = laplassGrad;

        DoubleMatrix* residuals = CalculateResidual(*p);
        RenewBounds(*residuals);

        DoubleMatrix* laplassResiduals = approximateOperations.CalculateLaplass(*residuals);
        RenewBounds(*laplassResiduals);

        DoubleMatrix* previousGrad = grad;
        grad = CalculateGradient(*residuals, *laplassResiduals, *previousGrad, *laplassPreviousGrad, iteration);
        delete previousGrad;
        delete laplassResiduals;

        laplassGrad = approximateOperations.CalculateLaplass(*grad);
        RenewBounds(*laplassGrad);

        double tau = CalculateTauValue(*residuals, *grad, *laplassGrad);
        delete residuals;
        delete previousP;
        previousP = p;
        p = CalculateNewP(*previousP, *grad, tau);
        ++iteration;
    }
    if (processorData.IsMainProcessor())
    {
        std::cout << "**************** last iteration = " << iteration << ", error = " << errorValue << std::endl;
    }
    delete grad;
    delete previousP;
    delete laplassGrad;
    delete laplassPreviousGrad;
    *error = errorValue;
    return p;
}

double ConjugateGradientAlgo::CalculateTauValue(const DoubleMatrix& residuals, const DoubleMatrix& grad, const DoubleMatrix& laplassGrad)
{
    double numerator = approximateOperations.ScalarProduct(residuals, grad);
    double denominator = approximateOperations.ScalarProduct(laplassGrad, grad);
    double tauValue = GetFractionValueFromAllProcessors(numerator, denominator);
    return tauValue;
}

double ConjugateGradientAlgo::CalculateAlphaValue(const DoubleMatrix& laplassResiduals, const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad)
{
    double numerator = approximateOperations.ScalarProduct(laplassResiduals, previousGrad);
    double denominator = approximateOperations.ScalarProduct(laplassPreviousGrad, previousGrad);
    double alphaValue = GetFractionValueFromAllProcessors(numerator, denominator);
    return alphaValue;
}

DoubleMatrix* ConjugateGradientAlgo::CalculateResidual(const DoubleMatrix& p)
{
    DoubleMatrix* laplassP = approximateOperations.CalculateLaplass(p);
    DoubleMatrix* residuals = new DoubleMatrix(laplassP->rowsCount(), laplassP->colsCount());
#ifdef PARALLEL
    #pragma omp parallel for
#endif
    for (int i = processorData.FirstOwnRowRelativeIndex(); i <= processorData.LastOwnRowRelativeIndex(); ++i)
    {
        for (int j = processorData.FirstOwnColRelativeIndex(); j <= processorData.LastOwnColRelativeIndex(); ++j)
        {
            bool isInnerPoint = processorData.IsInnerIndices(i, j);
            if (isInnerPoint)
            {
                (*residuals)(i, j) = 0;
            }
            else
            {
                (*residuals)(i, j) = (*laplassP)(i, j) - diffModel.CalculateFunctionValue(netModel.xValue(j), netModel.yValue(i));
            }
        }
    }
    delete laplassP;
    return residuals;
}

DoubleMatrix* ConjugateGradientAlgo::CalculateGradient(const DoubleMatrix& residuals, const DoubleMatrix& laplassResiduals,
                                const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad,
                                int k)
{
    DoubleMatrix* gradient;
    if (k == 0)
    {
        gradient = new DoubleMatrix(residuals);
    }
    else
    {
        double alpha = CalculateAlphaValue(laplassResiduals, previousGrad, laplassPreviousGrad);
        DoubleMatrix* tmp = alpha * previousGrad;
        gradient = residuals - *tmp;
        delete tmp;
    }
    return gradient;
}

DoubleMatrix* ConjugateGradientAlgo::CalculateNewP(const DoubleMatrix& p, const DoubleMatrix& grad, double tau)
{
    DoubleMatrix* tmp = tau * grad;
    DoubleMatrix* newValues = p - *tmp;
    delete tmp;
    return newValues;
}

double ConjugateGradientAlgo::CalculateError(const DoubleMatrix& uValues, const DoubleMatrix& p)
{
    DoubleMatrix* pCropped = p.CropMatrix(processorData.FirstOwnRowRelativeIndex(), processorData.RowsCount(),
                                 processorData.FirstOwnColRelativeIndex(), processorData.ColsCount());
    DoubleMatrix* psi = uValues - *pCropped;
    double error = approximateOperations.MaxNormValue(*psi);
    double globalError = GetMaxValueFromAllProcessors(error);
    delete pCropped;
    delete psi;
    return globalError;
}

bool ConjugateGradientAlgo::IsStopCondition(const DoubleMatrix& p, const DoubleMatrix& previousP)
{
    DoubleMatrix* pDiff = p - previousP;
    DoubleMatrix* pDiffCropped = pDiff->CropMatrix(processorData.FirstOwnRowRelativeIndex(), processorData.RowsCount(),
                                          processorData.FirstOwnColRelativeIndex(), processorData.ColsCount());
    double pDiffNormLocal = approximateOperations.MaxNormValue(*pDiffCropped);
    double pDiffNormGlobal = GetMaxValueFromAllProcessors(pDiffNormLocal);
    delete pDiff;
    delete pDiffCropped;
    return pDiffNormGlobal < eps;
}
