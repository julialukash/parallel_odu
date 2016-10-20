#include <iostream>
#include "Interface.h"
#include "ConjugateGradientAlgo.h"
#include "ApproximateOperations.h"

const double xMinBoundary = 0;
const double xMaxBoundary = 2;
const double yMinBoundary = 0;
const double yMaxBoundary = 2;
const int pointsCount = 1000;

int main(int argc, char *argv[])
{
    std::cout << "hello" << std::endl;
    auto diffEquation = new DifferentialEquationModel();
    auto netModel = new NetModel(xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary, pointsCount, pointsCount);
    auto netModelPtr = std::make_shared<NetModel>(*netModel);
//    std::cout << netModelPtr->xValue(0) << " " << netModelPtr->yValue(0) << std::endl;
//    std::cout << netModelPtr->xValue(pointsCount) << " " << netModelPtr->yValue(pointsCount) << std::endl;
    auto diffEquationPtr = std::make_shared<DifferentialEquationModel>(*diffEquation);
    auto derivator = new ApproximateOperations(netModelPtr);
    auto derivatorPtr = std::make_shared<ApproximateOperations>(*derivator);
    auto optimizationAlgo = new ConjugateGradientAlgo(netModelPtr, diffEquationPtr, derivatorPtr);

    optimizationAlgo->Process();
    // to do: save to file
}
