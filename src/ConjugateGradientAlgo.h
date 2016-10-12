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
public:
    void Init(DifferentialEquationModel model, NetModel netModel)
    {
        base_generator_type generator(198);
        boost::uniform_real<> xUniDistribution(netModel.xMinBoundary, netModel.xMaxBoundary);
        boost::uniform_real<> yUniDistribution(netModel.yMinBoundary, netModel.yMaxBoundary);
        boost::variate_generator<base_generator_type&, boost::uniform_real<> > xUniform(generator, xUniDistribution);
        boost::variate_generator<base_generator_type&, boost::uniform_real<> > yUniform(generator, yUniDistribution);

        for (auto i = 0; i < netModel.xSize(); ++i)
        {
            for (auto j = 0; j < netModel.ySize(); ++j)
            {
                if ((i == 0 || i == netModel.xSize()) && (j == 0 || j == netModel.ySize()))
                {
                    netModel[i, j] = model.CalculateBoundaryValue(netModel.xValue(i), netModel.yValue(j));
                }
                else
                {
                    // random init
                    netModel[i, j] = model.CalculateFunctionValue(xUniform(), yUniform());
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
