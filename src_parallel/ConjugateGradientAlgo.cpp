#include <iostream>
#include <mpi.h>

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

DoubleMatrix ConjugateGradientAlgo::CalculateU()
{
    auto values = DoubleMatrix(processorData->RowsCount(), netModel->yPointsCount);
    for (auto i = 0; i < values.rowsCount(); ++i)
    {
        auto iNetIndex = i + processorData->FirstRowIndex();
        for (auto j = 0; j < values.colsCount(); ++j)
        {
            values(i, j) = diffModel->CalculateUValue(netModel->xValue(iNetIndex), netModel->yValue(j));
//#ifdef DEBUG_MODE
//            std::cout << "i = " << i << ", j = " << j << ", iNetIndex = " << iNetIndex << std::endl;
//#endif
        }
    }

//#ifdef DEBUG_MODE
//    std::cout << values << std::endl;
//#endif
    return values;
}

DoubleMatrix ConjugateGradientAlgo::Init()
{
    auto values = DoubleMatrix(processorData->RowsCountWithBorders(), netModel->xPointsCount);
    for (auto i = 0; i < values.rowsCount(); ++i)
    {
        auto iValueIndex = processorData->IsFirstProcessor() ? i + 1 : i;
        auto iNetIndex = i + processorData->FirstRowWithBordersIndex();
        if (iNetIndex  >= processorData->FirstRowIndex() && iNetIndex <= processorData->LastRowIndex())
        {
            for (auto j = 0; j < values.colsCount(); ++j)
            {
                if (netModel->IsInnerPoint(iNetIndex, j))
                {
                    values(iValueIndex, j) = diffModel->CalculateBoundaryValue(netModel->xValue(iNetIndex), netModel->yValue(j));
                }
                else
                {
                    values(iValueIndex, j) = diffModel->CalculateFunctionValue(netModel->xValue(iNetIndex), netModel->yValue(j));
                }
//#ifdef DEBUG_MODE
//            std::cout << "i = " << i << ", j = " << j << ", iNetIndex = " << iNetIndex << std::endl;
//#endif
            }
        }
    }
//#ifdef DEBUG_MODE
//    std::cout << values << std::endl;
//#endif
    return values;
}


void ConjugateGradientAlgo::RenewBoundRows(DoubleMatrix& values)
{
//#ifdef DEBUG_MODE
//    std::cout << "RenewBoundRows \n" << values << std::endl;
//#endif
    MPI_Status status;
    int nextProcessorRank = processorData->IsLastProcessor() ? MPI_PROC_NULL : processorData->rank + 1;
    int previousProcessorRank = processorData->IsFirstProcessor() ? MPI_PROC_NULL : processorData->rank - 1;

//#ifdef DEBUG_MODE
//    std::cout << "nextProcessorRank = " << nextProcessorRank << ", previousProcessorRank = " << previousProcessorRank << std::endl;
//#endif

    // send to next processor last "no border" line
    // receive from prev processor first "border" line
    MPI_Sendrecv(&(values.matrix[processorData->RowsCountWithBorders() - 2][0]), netModel->xPointsCount, MPI_DOUBLE, nextProcessorRank, UP,
                 &(values.matrix[0][0]), netModel->xPointsCount, MPI_DOUBLE, previousProcessorRank, UP,
                 MPI_COMM_WORLD, &status);
    // send to prev processor first "no border" line
    // receive from next processor last "border" line
    MPI_Sendrecv(&(values.matrix[1][0]), netModel->xPointsCount, MPI_DOUBLE, previousProcessorRank, DOWN,
                 &(values.matrix[processorData->RowsCountWithBorders() - 1][0]), netModel->xPointsCount, MPI_DOUBLE, nextProcessorRank, DOWN,
                 MPI_COMM_WORLD, &status);

//#ifdef DEBUG_MODE
//    std::cout << "Finished RenewBoundRows \n" << values << std::endl;
//#endif
}


void ConjugateGradientAlgo::Process(DoubleMatrix &p, const DoubleMatrix& uValues)
{
    DoubleMatrix previousP, grad, laplassGrad, laplassPreviousGrad;
    int iteration = 0;
    double error = .0;
    while (true)
    {
#ifdef DEBUG_MODE
        std::cout << "rank = " << processorData->rank << " iteration = " << iteration << std::endl;
#endif
        error = CalculateError(uValues, p);
#ifdef DEBUG_MODE
        std::cout << "===================================================================" << std::endl;
        std::cout << "iteration = " << iteration << ", error = " << error << std::endl;
        std::cout << "p = " << p << std::endl;
#endif

        RenewBoundRows(p);
#ifdef DEBUG_MODE
        std::cout << "renewed p = \n" << p << std::endl;
#endif

        // check stop condition
        auto stopCondition = iteration != 0 && IsStopCondition(p, previousP);
        if (stopCondition)
        {
            p = p.CropMatrix(p, 1, p.rowsCount() - 2);
            break;
        }

        laplassPreviousGrad = laplassGrad;

#ifdef DEBUG_MODE
        std::cout << "//laplassPreviousGrad = \n" << laplassPreviousGrad << std::endl;
#endif

        auto residuals = CalculateResidual(p);
        RenewBoundRows(residuals);

        auto laplassResiduals = approximateOperations->CalculateLaplass(residuals, processorData);
        RenewBoundRows(laplassResiduals);

#ifdef DEBUG_MODE
        std::cout << "Residuals = \n" << residuals << std::endl;
        std::cout << "Laplass Residuals = " << laplassResiduals << std::endl;
        std::cout << "grad = " << grad << std::endl;
        std::cout << "laplassPreviousGrad = " << laplassPreviousGrad << std::endl;
#endif

        grad = CalculateGradient(residuals, laplassResiduals, grad, laplassPreviousGrad, iteration);

        laplassGrad = approximateOperations->CalculateLaplass(grad, processorData);
        RenewBoundRows(laplassGrad);

        auto tau = CalculateTauValue(residuals, grad, laplassGrad);

#ifdef DEBUG_MODE
        std::cout << "p = \n" << p << std::endl;
        std::cout << "grad = \n" << grad << std::endl;
        std::cout << "laplassGrad = \n" << laplassGrad << std::endl;
        std::cout << "tau = " << tau << std::endl;
        std::cout << "previousP = \n" << previousP << std::endl;
#endif

        previousP = p;
        p = CalculateNewP(p, grad, tau);

        ++iteration;        
    }
    return;
}


double ConjugateGradientAlgo::CalculateTauValue(const DoubleMatrix& residuals, const DoubleMatrix& grad, const DoubleMatrix& laplassGrad)
{    
#ifdef DEBUG_MODE
    std::cout << "CalculateTauValue" << std::endl;
#endif
    auto numerator = approximateOperations->ScalarProduct(residuals, grad, processorData);
    auto denominator = approximateOperations->ScalarProduct(laplassGrad, grad, processorData);
    double localTau[2] = {numerator, denominator};
    double globalTau[2] = {0, 0};
    MPI_Allreduce(localTau, globalTau, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef DEBUG_MODE
    std::cout << "CalculateTauValue num  = " << numerator << ", " << denominator
              << ", local tau = " << *localTau  << " " << *(localTau + 1)
              << ", global = " << *globalTau << " " <<  *(globalTau + 1) << std::endl;
//              << "tau = " << globalTau[0] / globalTau[1] << std::endl;
#endif
    auto tauValue = globalTau[0] / globalTau[1];
    return tauValue;
}

double ConjugateGradientAlgo::CalculateAlphaValue(const DoubleMatrix& laplassResiduals, const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad)
{
#ifdef DEBUG_MODE
    std::cout << "CalculateAlphaValue" << std::endl;
#endif
    auto numerator = approximateOperations->ScalarProduct(laplassResiduals, previousGrad, processorData);
    auto denominator = approximateOperations->ScalarProduct(laplassPreviousGrad, previousGrad, processorData);
    double localAlpha[2] = {numerator, denominator};
    double globalAlpha[2] = {0, 0};
    MPI_Allreduce(localAlpha, globalAlpha, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef DEBUG_MODE
    std::cout << "CalculateAlphaValue num  = " << numerator << ", " << denominator
              << ", localAlpha = " << *localAlpha  << " " << *(localAlpha + 1)
              << ", globalAlpha = " << *globalAlpha << " " <<  *(globalAlpha + 1) << std::endl;
//              << "alpha = " << globalAlpha[0] / globalAlpha[1] << std::endl;
#endif
    auto alphaValue = globalAlpha[0] / globalAlpha[1];
    return alphaValue;
}

DoubleMatrix ConjugateGradientAlgo::CalculateResidual(const DoubleMatrix& p)
{
#ifdef DEBUG_MODE
        std::cout << "CalculateResidual ..." << std::endl;
#endif
    auto laplassP = approximateOperations->CalculateLaplass(p, processorData);
#ifdef DEBUG_MODE
        std::cout << "laplassP \n" << laplassP << std::endl;
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
//#ifdef DEBUG_MODE
//        std::cout << "i = " << i << ", j = " << j << ", iNet = "
//                  << iNetIndex << " " << netModel->IsInnerPoint(iNetIndex, j)
//                  << ", laplassP(i, j) = " << laplassP(i, j)
//                  << ", value = " << diffModel->CalculateFunctionValue(netModel->xValue(j), netModel->yValue(iNetIndex))
//                  << ", residuals(i, j) = " << residuals(i, j) << std::endl;
//#endif
        }
    }
//#ifdef DEBUG_MODE
//        std::cout << "residuals \n" << residuals << std::endl;
//#endif
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

DoubleMatrix ConjugateGradientAlgo::CalculateNewP(const DoubleMatrix& p, const DoubleMatrix& grad, double tau)
{

#ifdef DEBUG_MODE
        std::cout << "CalculateNewP..." << std::endl;
#endif

    return p - tau * grad;
}

double ConjugateGradientAlgo::CalculateError(const DoubleMatrix& uValues, const DoubleMatrix& p)
{
#ifdef DEBUG_MODE
    std::cout << "CalculateError..." << std::endl;
#endif

    auto pCropped = p.CropMatrix(p, 1, p.rowsCount() - 2);
//#ifdef DEBUG_MODE
//    std::cout << "pCropped = \n" << pCropped << std::endl;
//#endif
    auto psi = uValues - pCropped;

//#ifdef DEBUG_MODE
//    std::cout << "psi = \n" << psi << std::endl;
//#endif

    auto error = approximateOperations->MaxNormValue(psi);

#ifdef DEBUG_MODE
    std::cout << "error = \n" << error << std::endl;
#endif
    return error;
}

bool ConjugateGradientAlgo::IsStopCondition(const DoubleMatrix& p, const DoubleMatrix& previousP)
{
    auto pDiff = p - previousP;
    auto pDiffCropped = pDiff.CropMatrix(pDiff, 1, pDiff.rowsCount() - 2);
    double pDiffNormLocal, pDiffNormGlobal;
    pDiffNormLocal = approximateOperations->MaxNormValue(pDiffCropped);
    MPI_Allreduce(&pDiffNormLocal, &pDiffNormGlobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

#ifdef DEBUG_MODE
    std::cout << "pDiffNormLocal = " << pDiffNormLocal << ", pDiffNormGlobal = " << pDiffNormGlobal << std::endl;

    auto stop = pDiffNormGlobal < eps;
    std::cout << "CheckStopCondition = " << stop << std::endl;
#endif
    return pDiffNormGlobal < eps;
}
