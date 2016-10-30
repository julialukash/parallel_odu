#include <iostream>

#include "ConjugateGradientAlgo.h"

#include "MPIOperations.h"
#define DEBUG_MODE = 1

ConjugateGradientAlgo::ConjugateGradientAlgo(std::shared_ptr<NetModel> model, std::shared_ptr<DifferentialEquationModel> modelDiff,
                  std::shared_ptr<ApproximateOperations> approximateOperationsPtr,
                  std::shared_ptr<ProcessorsData> processorDataPtr)
{
    netModel = model;
    diffModel = modelDiff;
    approximateOperations = approximateOperationsPtr;
    processorData = processorDataPtr;
}

std::shared_ptr<DoubleMatrix> ConjugateGradientAlgo::CalculateU()
{
    auto values = std::make_shared<DoubleMatrix>(processorData->RowsCount(), netModel->yPointsCount);
    for (auto i = 0; i < values->rowsCount(); ++i)
    {
        auto iNetIndex = i + processorData->FirstRowIndex();
        for (auto j = 0; j < values->colsCount(); ++j)
        {
            (*values)(i, j) = diffModel->CalculateUValue(netModel->xValue(iNetIndex), netModel->yValue(j));
        }
    }
    return values;
}

std::shared_ptr<DoubleMatrix> ConjugateGradientAlgo::Init()
{
    auto values = std::make_shared<DoubleMatrix>(processorData->RowsCountWithBorders(), netModel->xPointsCount);
    for (auto i = 0; i < values->rowsCount(); ++i)
    {
        auto iValueIndex = processorData->IsFirstProcessor() ? i + 1 : i;
        auto iNetIndex = i + processorData->FirstRowWithBordersIndex();
        if (iNetIndex  >= processorData->FirstRowIndex() && iNetIndex <= processorData->LastRowIndex())
        {
            for (auto j = 0; j < values->colsCount(); ++j)
            {
                if (netModel->IsInnerPoint(iNetIndex, j))
                {
                    (*values)(iValueIndex, j) = diffModel->CalculateBoundaryValue(netModel->xValue(iNetIndex), netModel->yValue(j));
                }
                else
                {
                    (*values)(iValueIndex, j) = diffModel->CalculateFunctionValue(netModel->xValue(iNetIndex), netModel->yValue(j));
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

double ConjugateGradientAlgo::Process(std::shared_ptr<DoubleMatrix> p, const DoubleMatrix& uValues)
{
    std::shared_ptr<DoubleMatrix> previousP = p;
    DoubleMatrix grad, laplassGrad, laplassPreviousGrad;
    int iteration = 0;
    double error = -1.0;
    while (true)
    {
#ifdef DEBUG_MODE
        std::cout << "Process u vals = \n" << uValues << std::endl;
        std::cout << "Process rank = " << processorData->rank << " iteration = " << iteration << std::endl;
        std::cout << "===================================================================" << std::endl;
        std::cout << "Process iteration = " << iteration << ", error = " << error << std::endl;
        std::cout << "Process p = " << *p << std::endl;
#endif

        RenewBoundRows(*p);
#ifdef DEBUG_MODE
        std::cout << "Process renewed p = \n" << *p << std::endl;
#endif

        error = CalculateError(uValues, *p);

        // check stop condition
        auto stopCondition = iteration != 0 && IsStopCondition(*p, *previousP);
        if (stopCondition)
        {
            p = p->CropMatrix(1, p->rowsCount() - 2);
            break;
        }

        laplassPreviousGrad = laplassGrad;

#ifdef DEBUG_MODE
        std::cout << "Process laplassPreviousGrad = \n" << laplassPreviousGrad << std::endl;
#endif

        auto residuals = CalculateResidual(*p);
        RenewBoundRows(residuals);
#ifdef DEBUG_MODE
        std::cout << "Process Residuals = \n" << residuals << std::endl;
#endif
        auto laplassResiduals = approximateOperations->CalculateLaplass(residuals, processorData);
        RenewBoundRows(laplassResiduals);

#ifdef DEBUG_MODE
//        std::cout << "Residuals = \n" << residuals << std::endl;
        std::cout << "Process Laplass Residuals = " << laplassResiduals << std::endl;
        std::cout << "Process grad = " << grad << std::endl;
        std::cout << "Process laplassPreviousGrad = " << laplassPreviousGrad << std::endl;
#endif

        grad = CalculateGradient(residuals, laplassResiduals, grad, laplassPreviousGrad, iteration);

        laplassGrad = approximateOperations->CalculateLaplass(grad, processorData);
        RenewBoundRows(laplassGrad);

        auto tau = CalculateTauValue(residuals, grad, laplassGrad);

#ifdef DEBUG_MODE
        std::cout << "Process p = \n" << *p << std::endl;
        std::cout << "Process grad = \n" << grad << std::endl;
        std::cout << "Process laplassGrad = \n" << laplassGrad << std::endl;
        std::cout << "Process tau = " << tau << std::endl;
        std::cout << "Process previousP = \n" << *previousP << std::endl;
        std::cout << "Process ... " << std::endl;
#endif

        previousP.swap(p);
        p = CalculateNewP(*previousP, grad, tau);

        ++iteration;        
    }

#ifdef DEBUG_MODE
        std::cout << "**************** last iteration = " << iteration << ", error = " << error << std::endl;
#endif
    return error;
}


double ConjugateGradientAlgo::CalculateTauValue(const DoubleMatrix& residuals, const DoubleMatrix& grad, const DoubleMatrix& laplassGrad)
{    
#ifdef DEBUG_MODE
    std::cout << "CalculateTauValue residuals\n " << residuals << std::endl;
    std::cout << "CalculateTauValue laplassGrad\n " << laplassGrad << std::endl;
    std::cout << "CalculateTauValue grad\n " << grad << std::endl;
#endif
    auto numerator = approximateOperations->ScalarProduct(residuals, grad, processorData);
    auto denominator = approximateOperations->ScalarProduct(laplassGrad, grad, processorData);
    auto tauValue = getFractionValueFromAllProcessors(numerator, denominator);    
#ifdef DEBUG_MODE
    std::cout << "CalculateTauValue tauValue = " << tauValue << std::endl;
#endif
    return tauValue;
}

double ConjugateGradientAlgo::CalculateAlphaValue(const DoubleMatrix& laplassResiduals, const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad)
{
#ifdef DEBUG_MODE
    std::cout << "CalculateAlphaValue laplassResiduals\n " << laplassResiduals << std::endl;
    std::cout << "CalculateAlphaValue laplassPreviousGrad\n " << laplassPreviousGrad << std::endl;
    std::cout << "CalculateAlphaValue previousGrad\n " << previousGrad << std::endl;
#endif
    auto numerator = approximateOperations->ScalarProduct(laplassResiduals, previousGrad, processorData);
    auto denominator = approximateOperations->ScalarProduct(laplassPreviousGrad, previousGrad, processorData);
    auto alphaValue = getFractionValueFromAllProcessors(numerator, denominator);
#ifdef DEBUG_MODE
    std::cout << "CalculateAlphaValue alphaValue = " << alphaValue << std::endl;
#endif
    return alphaValue;
}

DoubleMatrix ConjugateGradientAlgo::CalculateResidual(const DoubleMatrix& p)
{
#ifdef DEBUG_MODE
        std::cout << "CalculateResidual ..." << std::endl;
#endif
    auto laplassP = approximateOperations->CalculateLaplass(p, processorData);
#ifdef DEBUG_MODE
        std::cout << "CalculateResidual laplassP \n" << laplassP << std::endl;
#endif
    auto residuals = DoubleMatrix(laplassP.rowsCount(), laplassP.colsCount());
    auto startIndex = 1;
    auto endIndex = processorData->RowsCountWithBorders() - 2;
    for (auto i = startIndex; i <= endIndex; ++i)
    {
        auto iNetIndex = i - startIndex + processorData->FirstRowIndex();
        for (auto j = 0; j < residuals.colsCount(); ++j)
        {
            if (netModel->IsInnerPoint(iNetIndex, j))
            {
                residuals(i, j) = 0;
            }
            else
            {
                residuals(i, j) = laplassP(i, j) - diffModel->CalculateFunctionValue(netModel->xValue(j), netModel->yValue(iNetIndex));
            }
        }
    }
    return residuals;
}

DoubleMatrix ConjugateGradientAlgo::CalculateGradient(const DoubleMatrix& residuals, const DoubleMatrix& laplassResiduals,
                                const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad,
                                int k)
{
#ifdef DEBUG_MODE
        std::cout << "CalculateGradient = \n" << residuals << std::endl;
#endif
    DoubleMatrix gradient;
    if (k == 0)
    {
        gradient = residuals;
    }
    else
    {
        auto alpha = CalculateAlphaValue(laplassResiduals, previousGrad, laplassPreviousGrad);
#ifdef DEBUG_MODE
        std::cout << "Alpha = " << alpha << std::endl;
#endif
        gradient = residuals - alpha * laplassResiduals;
    }
    return gradient;
}

std::shared_ptr<DoubleMatrix> ConjugateGradientAlgo::CalculateNewP(const DoubleMatrix& p, const DoubleMatrix& grad, double tau)
{
#ifdef DEBUG_MODE
    std::cout << "CalculateNewP p = \n" << p << std::endl;
    std::cout << "CalculateNewP grad = \n" << grad << std::endl;
    std::cout << "CalculateNewP tau = " << tau << std::endl;
#endif
    return std::make_shared<DoubleMatrix>(p - tau * grad);
}

double ConjugateGradientAlgo::CalculateError(const DoubleMatrix& uValues, const DoubleMatrix& p)
{
#ifdef DEBUG_MODE
    std::cout << "CalculateError..." << std::endl;
    std::cout << "p = \n" << p << std::endl;
#endif

    auto pCropped = p.CropMatrix(1, p.rowsCount() - 2);
#ifdef DEBUG_MODE
    std::cout << "uValues = \n" << uValues << std::endl;
    std::cout << "pCropped = \n" << *pCropped << std::endl;
#endif
    auto psi = uValues - *pCropped;

#ifdef DEBUG_MODE
    std::cout << "psi = \n" << psi << std::endl;
#endif
    auto error = approximateOperations->MaxNormValue(psi);
    auto globalError = getMaxValueFromAllProcessors(error);

#ifdef DEBUG_MODE
    std::cout << "error = " << error << ", global = " << globalError << std::endl;
#endif
    return globalError;
}

bool ConjugateGradientAlgo::IsStopCondition(const DoubleMatrix& p, const DoubleMatrix& previousP)
{
#ifdef DEBUG_MODE
    std::cout << "IsStopCondition p = \n" << p << std::endl;
    std::cout << "IsStopCondition previousP = \n" << previousP << std::endl;
#endif

    auto pDiff = p - previousP;
    auto pDiffCropped = pDiff.CropMatrix(1, pDiff.rowsCount() - 2);
#ifdef DEBUG_MODE
    std::cout << "IsStopCondition pDiff = \n" << pDiff << std::endl;
    std::cout << "IsStopCondition pDiffCropped = \n" << pDiffCropped << std::endl;
#endif
    double pDiffNormLocal, pDiffNormGlobal;
    pDiffNormLocal = approximateOperations->MaxNormValue(*pDiffCropped);
    pDiffNormGlobal = getMaxValueFromAllProcessors(pDiffNormLocal);

#ifdef DEBUG_MODE
    std::cout << "pDiffNormLocal = " << pDiffNormLocal << ", pDiffNormGlobal = " << pDiffNormGlobal << std::endl;

    auto stop = pDiffNormGlobal < eps;
    std::cout << "CheckStopCondition = " << stop << std::endl;
#endif
    return pDiffNormGlobal < eps;
}
