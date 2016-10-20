#ifndef CONJUGATEGRADIENTALGO_H
#define CONJUGATEGRADIENTALGO_H

#include "interface.h"

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
public:
    ConjugateGradient(std::shared_ptr<NetModel> model, std::shared_ptr<DifferentialEquationModel> modelDiff)
    {
        netModel = model;
        diffModel = modelDiff;
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


    double CalculateTauValue()
    {
        return 0;
    }

    double CalculateAlphaValue()
    {
        return 0;
    }


};

#endif // CONJUGATEGRADIENTALGO_H
