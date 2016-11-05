#include <iostream>

#include "conjugate_gradient_algo.h"
#include "mpi_operations.h"

#define DEBUG_MODE = 1

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
                    (*values)(iValueIndex, j) = 0.05;
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
#ifdef DEBUG_MODE
        std::cout << "Process u vals = \n" << uValues << std::endl;
        std::cout << "Process rank = " << processorData.rank << " iteration = " << iteration << std::endl;
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
            auto pCroppedPtr = p->CropMatrix(1, p->rowsCount() - 2);
//            p = std::make_shared<DoubleMatrix>(*pCroppedPtr);
            p.reset();
            p = pCroppedPtr;

#ifdef DEBUG_MODE
//            std::cout << "Process! break p  = \n" << *p << std::endl;
#endif
            break;
        }

        laplassPreviousGrad.swap(laplassGrad);

#ifdef DEBUG_MODE
        std::cout << "Process laplassPreviousGrad = \n" << *laplassPreviousGrad << std::endl;
#endif

        auto residuals = CalculateResidual(*p);
        RenewBoundRows(*residuals);
#ifdef DEBUG_MODE
        std::cout << "Process Residuals = \n" << *residuals << std::endl;
#endif
        auto laplassResiduals = approximateOperations.CalculateLaplass(*residuals);
        RenewBoundRows(*laplassResiduals);

#ifdef DEBUG_MODE
        std::cout << "Process Laplass Residuals = " << *laplassResiduals << std::endl;
        std::cout << "Process grad = " << *grad << std::endl;
        std::cout << "Process laplassPreviousGrad = " << *laplassPreviousGrad << std::endl;
#endif

        grad = CalculateGradient(residuals, *laplassResiduals, *grad, *laplassPreviousGrad, iteration);

#ifdef DEBUG_MODE
        std::cout << "Process CalculateGradient finished" << std::endl;
#endif
        laplassGrad = approximateOperations.CalculateLaplass(*grad);

#ifdef DEBUG_MODE
        std::cout << "Process CalculateLaplass finished" << std::endl;
#endif
        RenewBoundRows(*laplassGrad);
#ifdef DEBUG_MODE
    std::cout << "Process RenewBoundRows finished" << std::endl;
#endif
        auto tau = CalculateTauValue(*residuals, *grad, *laplassGrad);

#ifdef DEBUG_MODE
        std::cout << "Process p = \n" << *p << std::endl;
        std::cout << "Process grad = \n" << *grad << std::endl;
        std::cout << "Process laplassGrad = \n" << *laplassGrad << std::endl;
        std::cout << "Process tau = " << tau << std::endl;
        std::cout << "Process previousP = \n" << *previousP << std::endl;
        std::cout << "Process ... " << std::endl;
#endif
        previousP.swap(p);
        p = CalculateNewP(*previousP, *grad, tau);

#ifdef DEBUG_MODE
        std::cout << "CalculateNewP finished" << std::endl;
#endif
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
    double numerator = approximateOperations.ScalarProduct(residuals, grad);
    double denominator = approximateOperations.ScalarProduct(laplassGrad, grad);
    double tauValue = GetFractionValueFromAllProcessors(numerator, denominator);
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
    double numerator = approximateOperations.ScalarProduct(laplassResiduals, previousGrad);
    double denominator = approximateOperations.ScalarProduct(laplassPreviousGrad, previousGrad);
    double alphaValue = GetFractionValueFromAllProcessors(numerator, denominator);
#ifdef DEBUG_MODE
    std::cout << "CalculateAlphaValue alphaValue = " << alphaValue << std::endl;
#endif
    return alphaValue;
}

std::shared_ptr<DoubleMatrix> ConjugateGradientAlgo::CalculateResidual(const DoubleMatrix& p)
{
#ifdef DEBUG_MODE
        std::cout << "CalculateResidual ..." << std::endl;
#endif
    auto laplassP = approximateOperations.CalculateLaplass(p);
#ifdef DEBUG_MODE
        std::cout << "CalculateResidual laplassP \n" << *laplassP << std::endl;
#endif
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
#ifdef DEBUG_MODE
        std::cout << "CalculateGradient = \n" << residuals << std::endl;
#endif
    std::shared_ptr<DoubleMatrix> gradient;
    if (k == 0)
    {
        gradient = residuals;
    }
    else
    {
        double alpha = CalculateAlphaValue(laplassResiduals, previousGrad, laplassPreviousGrad);
#ifdef DEBUG_MODE
        std::cout << "Alpha = " << alpha << std::endl;
#endif
        gradient = *residuals - *(alpha * previousGrad);
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
    return p - *(tau * grad);
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
    double error = approximateOperations.MaxNormValue(*psi);
    double globalError = GetMaxValueFromAllProcessors(error);

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
    auto pDiffCropped = pDiff->CropMatrix(1, pDiff->rowsCount() - 2);
#ifdef DEBUG_MODE
    std::cout << "IsStopCondition pDiff = \n" << *pDiff << std::endl;
    std::cout << "IsStopCondition pDiffCropped = \n" << *pDiffCropped << std::endl;
#endif
    double pDiffNormLocal, pDiffNormGlobal;
    pDiffNormLocal = approximateOperations.MaxNormValue(*pDiffCropped);
    pDiffNormGlobal = GetMaxValueFromAllProcessors(pDiffNormLocal);

#ifdef DEBUG_MODE
    std::cout << "pDiffNormLocal = " << pDiffNormLocal << ", pDiffNormGlobal = " << pDiffNormGlobal << std::endl;

    bool stop = pDiffNormGlobal < eps;
    std::cout << "CheckStopCondition = " << stop << std::endl;
#endif
    return pDiffNormGlobal < eps;
}
