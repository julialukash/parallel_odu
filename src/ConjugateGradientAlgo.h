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

class ConjugateGradient
{
private:
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

    void Init(double_matrix values)
    {
        base_generator_type generator(198);
        boost::uniform_real<> xUniDistribution(netModel->xMinBoundary, netModel->xMaxBoundary);
        boost::uniform_real<> yUniDistribution(netModel->yMinBoundary, netModel->yMaxBoundary);
        boost::variate_generator<base_generator_type&, boost::uniform_real<> > xUniform(generator, xUniDistribution);
        boost::variate_generator<base_generator_type&, boost::uniform_real<> > yUniform(generator, yUniDistribution);

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
            gradient = residuals - alpha * laplassResiduals;
        }
        return gradient;
    }

    bool CheckStopCondition()
    {
        return true;
    }

    void Process()
    {
        auto p = double_matrix(netModel->xPointsCount, netModel->yPointsCount);
        Init(p);
        auto grad = double_matrix(netModel->xPointsCount, netModel->yPointsCount);
        int iteration = 0;
        while (CheckStopCondition())
        {
            std::cout << "iteration = " << iteration << std::endl;
            ++iteration;
            auto residuals = CalculateResidual(p);
            auto laplassResiduals = derivator->CalculateLaplassApproximately(residuals);
        }
    }

};

#endif // CONJUGATEGRADIENTALGO_H
