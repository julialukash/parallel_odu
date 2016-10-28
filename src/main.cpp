#include <iostream>
#include <fstream>

#include "ApproximateOperations.h"
#include "ConjugateGradientAlgo.h"
#include "DifferentialEquationModel.h"
#include "Interface.h"

const double xMinBoundary = 0;
const double xMaxBoundary = 2;
const double yMinBoundary = 0;
const double yMaxBoundary = 2;


void writeValues(char* filename, const DoubleMatrix& values)
{
    std::ofstream outputFile(filename);
    if (!outputFile.is_open())
    {
        std::cerr << "Incorrect output file " << filename;
        exit(1);
    }

    for (auto i = 0; i < values.size1(); ++i)
    {
        for (auto j = 0; j < values.size2(); ++j)
        {
            outputFile << values(i,j) << " ";
        }
        outputFile << "\n";
    }

    outputFile.close();
}


int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Not enough input arguments\n";
        exit(1);
    }
    std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
    auto fileName = "out_main.txt";
    std::ofstream out(fileName);
    std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!

    auto groundValuesFilename = argv[1];
    auto approximateValuesFilename = argv[2];
    auto pointsCount = std::stoi(argv[3]);

    auto netModelPtr = std::make_shared<NetModel>(xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary, pointsCount, pointsCount);
    auto diffEquationPtr = std::make_shared<DifferentialEquationModel>();
    auto approximateOperationsPtr = std::make_shared<ApproximateOperations>(netModelPtr);
    auto optimizationAlgo = new ConjugateGradientAlgo(netModelPtr, diffEquationPtr, approximateOperationsPtr);

    //    std::cout << netModelPtr->xValue(0) << " " << netModelPtr->yValue(0) << std::endl;
    //    std::cout << netModelPtr->xValue(pointsCount) << " " << netModelPtr->yValue(pointsCount) << std::endl;

    auto uValues = diffEquationPtr->CalculateUValues(netModelPtr);
    std::cout << "u = \n" << uValues << std::endl;
    auto uValuesApproximate = optimizationAlgo->Init();
    std::cout << "p init = \n" << uValuesApproximate << std::endl;
    auto begin = omp_get_wtime();

    optimizationAlgo->Process(uValuesApproximate, uValues);

    auto time_elapsed = omp_get_wtime() - begin;


    std::cout.rdbuf(coutbuf); //reset to standard output again
    std::cout << "Elapsed time is " << time_elapsed << " sec" << std::endl;

    writeValues(groundValuesFilename, uValues);
    writeValues(approximateValuesFilename, uValuesApproximate);
}
