#include "ConjugateGradientAlgo.h"


#define DEBUG_MODE = 1


ConjugateGradientAlgo::ConjugateGradientAlgo(std::shared_ptr<NetModel> model, std::shared_ptr<DifferentialEquationModel> modelDiff,
                  std::shared_ptr<ApproximateOperations> approximateOperationsPtr)
{
#ifdef DEBUG_MODE
    std::cout << "Constructor  ConjugateGradientAlgo" << std::endl;
#endif

    netModel = model;
    diffModel = modelDiff;
    approximateOperations = approximateOperationsPtr;

#ifdef DEBUG_MODE
    std::cout << "Constructor finished ConjugateGradientAlgo" << std::endl;
#endif
}

DoubleMatrix ConjugateGradientAlgo::Init()
{
    auto values = DoubleMatrix(netModel->xPointsCount, netModel->yPointsCount);
    for (auto i = 0; i < values.size1(); ++i)
    {
        for (auto j = 0; j < values.size2(); ++j)
        {
            if (netModel->IsInnerPoint(i, j))
            {
                values(i, j) = diffModel->CalculateBoundaryValue(netModel->xValue(i), netModel->yValue(j));
            }
            else
            {
                // random init
                values(i, j) = diffModel->CalculateFunctionValue(netModel->xValue(i), netModel->yValue(j));
            }
        }
    }
#ifdef DEBUG_MODE
    std::cout << values << std::endl;
#endif
    return values;
}

void ConjugateGradientAlgo::Process(DoubleMatrix &p, const DoubleMatrix& uValues)
{
    DoubleMatrix previousP, grad, laplassGrad, laplassPreviousGrad;

    int iteration = 0;
    while (iteration == 0 || !IsStopCondition(p, previousP))
    {
        std::cout << "==============================================================" << std::endl;
        std::cout << "iteration = " << iteration << ", error = " << CalculateError(p, uValues) << std::endl;

#ifdef DEBUG_MODE
        std::cout << "p = \n" << p << std::endl;
#endif

        laplassPreviousGrad = laplassGrad;

        auto residuals = CalculateResidual(p);
        auto laplassResiduals = approximateOperations->CalculateLaplass(residuals);

#ifdef DEBUG_MODE
        std::cout << "Residuals = \n" << residuals << std::endl;
        std::cout << "Laplass Residuals = \n" << laplassResiduals << std::endl;
        std::cout << "grad = \n" << grad << std::endl;
        std::cout << "laplassPreviousGrad = \n" << laplassPreviousGrad << std::endl;
#endif

        grad = CalculateGradient(residuals, laplassResiduals, grad, laplassPreviousGrad, iteration);

        laplassGrad = approximateOperations->CalculateLaplass(grad);

        auto tau = CalculateTauValue(residuals, grad, laplassGrad);

#ifdef DEBUG_MODE
        std::cout << "grad = " << grad << std::endl;
        std::cout << "laplassGrad = " << laplassGrad << std::endl;
        std::cout << "tau = " << tau << std::endl;
        std::cout << "previousP = " << previousP << std::endl;
#endif

        previousP = p;
        p = CalculateNewP(p, grad, tau);

        ++iteration;
    }
}

double ConjugateGradientAlgo::CalculateTauValue(const DoubleMatrix& residuals, const DoubleMatrix& grad, const DoubleMatrix& laplassGrad)
{
    auto numerator = approximateOperations->ScalarProduct(residuals, grad);
    auto denominator = approximateOperations->ScalarProduct(laplassGrad, grad);
    auto tau = numerator / denominator;
    return tau;
}

double ConjugateGradientAlgo::CalculateAlphaValue(const DoubleMatrix& laplassResiduals, const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad)
{
    auto numerator = approximateOperations->ScalarProduct(laplassResiduals, previousGrad);
    auto denominator = approximateOperations->ScalarProduct(laplassPreviousGrad, previousGrad);
    auto alpha = numerator / denominator;
    return alpha;
}

DoubleMatrix ConjugateGradientAlgo::CalculateResidual(const DoubleMatrix& p)
{
    auto laplassP = approximateOperations->CalculateLaplass(p);
    auto residuals = DoubleMatrix(netModel->xPointsCount, netModel->yPointsCount);
    for (auto i = 0; i < residuals.size1(); ++i)
    {
        for (auto j = 0; j < residuals.size2(); ++j)
        {
            if (netModel->IsInnerPoint(i, j))
            {
                residuals(i, j) = 0;
            }
            else
            {
                residuals(i, j) = laplassP(i, j) - diffModel->CalculateFunctionValue(netModel->xValue(i), netModel->yValue(j));
            }
        }
    }
    return residuals;
}

DoubleMatrix ConjugateGradientAlgo::CalculateGradient(const DoubleMatrix& residuals, const DoubleMatrix& laplassResiduals,
                                const DoubleMatrix& previousGrad, const DoubleMatrix& laplassPreviousGrad,
                                int k)
{
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
    return p - tau * grad;
}

double ConjugateGradientAlgo::CalculateError(const DoubleMatrix& uValues, const DoubleMatrix& p)
{
    auto psi = uValues - p;
    auto error = approximateOperations->NormValue(psi);
    return error;
}

bool ConjugateGradientAlgo::IsStopCondition(const DoubleMatrix& p, const DoubleMatrix& previousP)
{
    auto pDiff = p - previousP;
    auto pDiffNorm = approximateOperations->NormValue(pDiff);
    std::cout << "pDiffNorm = " << pDiffNorm << std::endl;

#ifdef DEBUG_MODE
    auto stop = pDiffNorm < eps;
    std::cout << "CheckStopCondition = " << stop << std::endl;
#endif
    return pDiffNorm < eps;
}
