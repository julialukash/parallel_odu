#ifndef CONJUGATEGRADIENTALGO_H
#define CONJUGATEGRADIENTALGO_H

#include "interface.h"
#include "Derivator.h"

#include <boost/algorithm/minmax_element.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

typedef boost::minstd_rand base_generator_type;

//#define DEBUG_MODE = 1

class ConjugateGradient
{
private:
    const double eps = 10e-4;

    std::shared_ptr<NetModel> netModel;
    std::shared_ptr<DifferentialEquationModel> diffModel;
    std::shared_ptr<Derivator> derivator;
public:
    ConjugateGradient(std::shared_ptr<NetModel> model, std::shared_ptr<DifferentialEquationModel> modelDiff,
                      std::shared_ptr<Derivator> derivatorPtr)
    {
        netModel = model;
        diffModel = modelDiff;
        derivator = derivatorPtr;
    }

    double_matrix Init()
    {
        base_generator_type generator(198);
        boost::uniform_real<> xUniDistribution(netModel->xMinBoundary, netModel->xMaxBoundary);
        boost::uniform_real<> yUniDistribution(netModel->yMinBoundary, netModel->yMaxBoundary);
        boost::variate_generator<base_generator_type&, boost::uniform_real<> > xUniform(generator, xUniDistribution);
        boost::variate_generator<base_generator_type&, boost::uniform_real<> > yUniform(generator, yUniDistribution);

        auto values = double_matrix(netModel->xPointsCount, netModel->yPointsCount);
        for (size_t i = 0; i < values.size1(); ++i)
        {
            for (size_t j = 0; j < values.size2(); ++j)
            {
                if ((i == 0 || i == values.size1()) && (j == 0 || j == values.size2()))
                {
                    values(i, j) = diffModel->CalculateBoundaryValue(netModel->xValue(i), netModel->yValue(j));
                }
                else
                {
                    // random init
                    values(i, j) = diffModel->CalculateFunctionValue(xUniform(), yUniform());
                }
            }
        }
#ifdef DEBUG_MODE
        std::cout <<values<< std::endl;
#endif
        return values;
    }


    double CalculateTauValue(double_matrix residuals, double_matrix grad, double_matrix laplassGrad)
    {
        auto numerator = derivator->ScalarProduct(residuals, grad);
        auto denominator = derivator->ScalarProduct(-laplassGrad, grad);
        auto tau = numerator / denominator;
        return tau;
    }

    double CalculateAlphaValue(double_matrix laplassResiduals, double_matrix previousGrad, double_matrix laplassPreviousGrad)
    {
        auto numerator = derivator->ScalarProduct(-laplassResiduals, previousGrad);
        auto denominator = derivator->ScalarProduct(-laplassPreviousGrad, previousGrad);
        auto alpha = numerator / denominator;
        return alpha;
    }

    double_matrix CalculateResidual(double_matrix p)
    {
        auto laplassP = derivator->CalculateLaplassApproximately(p);
        auto residuals = double_matrix(netModel->xPointsCount, netModel->yPointsCount);
        for (size_t i = 0; i < residuals.size1(); ++i)
        {
            for (size_t j = 0; j < residuals.size2(); ++j)
            {
                if ((i == 0 || i == residuals.size1()) && (j == 0 || j == residuals.size2()))
                {
                    residuals(i, j) = 0;
                }
                else
                {
                    residuals(i, j) = -laplassP(i, j) - diffModel->CalculateFunctionValue(netModel->xValue(i), netModel->yValue(j));
                }
            }
        }
        return residuals;
    }

    double_matrix CalculateGradient(double_matrix residuals, double_matrix laplassResiduals,
                                    double_matrix previousGrad, double_matrix laplassPreviousGrad,
                                    int k)
    {
        double_matrix gradient;
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

    double_matrix CalculateNewP(double_matrix p, double_matrix grad, double tau)
    {
        return p - tau * grad;
    }

    bool IsStopCondition(double_matrix p, double_matrix previousP)
    {
        auto pDiff = p - previousP;
        auto pDiffNorm = derivator->NormValue(pDiff);
        std::cout << "pDiffNorm = " << pDiffNorm << std::endl;

#ifdef DEBUG_MODE
        auto stop = pDiffNorm < eps;
        std::cout << "CheckStopCondition = " << stop << std::endl;
#endif
        return pDiffNorm < eps;
    }

    void Process()
    {
        double_matrix previousP, grad, laplassGrad, laplassPreviousGrad;
//        grad = double_matrix(netModel->xPointsCount, netModel->yPointsCount);

        auto p = Init();

        int iteration = 0;
        while (iteration == 0 || !IsStopCondition(p, previousP))
        {
            std::cout << "iteration = " << iteration << std::endl;

#ifdef DEBUG_MODE
            std::cout << "p = " << p << std::endl;
#endif

            laplassPreviousGrad = laplassGrad;

            auto residuals = CalculateResidual(p);
            auto laplassResiduals = derivator->CalculateLaplassApproximately(residuals);

#ifdef DEBUG_MODE
            std::cout << "Residuals = " << residuals << std::endl;
            std::cout << "Laplass Residuals = " << laplassResiduals << std::endl;
            std::cout << "grad = " << grad << std::endl;
            std::cout << "laplassPreviousGrad = " << laplassPreviousGrad << std::endl;
#endif

            grad = CalculateGradient(residuals, laplassResiduals, grad, laplassPreviousGrad, iteration);

            laplassGrad = derivator->CalculateLaplassApproximately(grad);

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

};

#endif // CONJUGATEGRADIENTALGO_H
