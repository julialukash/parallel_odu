#include <iostream>
#include "interface.h"
#include "ConjugateGradientAlgo.h"
#include "Derivator.h"

const double xMinBoundary = 0;
const double xMaxBoundary = 2;
const double yMinBoundary = 0;
const double yMaxBoundary = 2;
const long pointsCount = 10;

int main(int argc, char *argv[])
{
    std::cout << "hello" << std::endl;
    auto diffEquation = new DifferentialEquationModel();
    auto netModel = new NetModel(xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary, pointsCount, pointsCount);
    auto netModelPtr = std::make_shared<NetModel>(*netModel);
    auto diffEquationPtr = std::make_shared<DifferentialEquationModel>(*diffEquation);
    auto gradient = new ConjugateGradient(netModelPtr, diffEquationPtr);
    auto derivator = new Derivator(netModelPtr);
}
