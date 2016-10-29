#include <iostream>
#include <fstream>
#include <mpi.h>

#include "ApproximateOperations.h"
#include "ConjugateGradientAlgo.h"
#include "DifferentialEquationModel.h"
#include "Interface.h"
#include "MPIOperations.h"
#include "ProcessorsData.h"

const double xMinBoundary = 0;
const double xMaxBoundary = 2;
const double yMinBoundary = 0;
const double yMaxBoundary = 2;
const double eps = 1e-4;

#define DEBUG_MAIN

void writeValues(char* filename, const DoubleMatrix& values)
{
    std::ofstream outputFile(filename);
    if (!outputFile.is_open())
    {
        std::cerr << "Incorrect output file " << filename;
        exit(1);
    }

    for (auto i = 0; i < values.rowsCount(); ++i)
    {
        for (auto j = 0; j < values.colsCount(); ++j)
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
    int rank, processorsCount;
    try
    {
        auto beginTime = clock();
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processorsCount);

        std::cout << "rank = " << rank << std::endl;

        auto processorInfoPtr = std::make_shared<ProcessorsData>(rank, processorsCount);

        std::cout << "(-) Processor with rank = " << processorInfoPtr->rank << std::endl;

        auto groundValuesFilename = argv[1];
        auto approximateValuesFilename = argv[2];
        auto pointsCount = std::stoi(argv[3]);

        auto netModelPtr = std::make_shared<NetModel>(xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary, pointsCount, pointsCount);
        auto diffEquationPtr = std::make_shared<DifferentialEquationModel>();
        auto approximateOperationsPtr = std::make_shared<ApproximateOperations>(netModelPtr);

        std::cout << "(*)Processor with rank = " << processorInfoPtr->rank << std::endl;
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        auto fileName = "out/out_rank" + std::to_string(processorInfoPtr->rank)  + ".txt";
        std::ofstream out(fileName);
        std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!

        std::cout << "(+)Processor with rank = " << processorInfoPtr->rank << std::endl;
        // init processors with their part of data
        processorInfoPtr->rowsCountValue = (netModelPtr->yPointsCount) / (processorInfoPtr->processorsCount);
        auto leftRowsCount = netModelPtr->yPointsCount - processorInfoPtr->rowsCountValue  * (processorInfoPtr->processorsCount);

        processorInfoPtr->startRowIndex = (processorInfoPtr->rank) * processorInfoPtr->rowsCountValue;
        if (leftRowsCount != 0 & processorInfoPtr->IsLastProcessor())
        {
            processorInfoPtr->rowsCountValue = processorInfoPtr->rowsCountValue + leftRowsCount;
        }
#ifdef DEBUG_MAIN
        std::cout << "Finished" << std::endl;
        std::cout << "rank = " << processorInfoPtr->rank << ", processorsCount = " << processorInfoPtr->processorsCount << std::endl
                  << "FirstRowIndex = " << processorInfoPtr->FirstRowIndex()
                  << ", LastRowIndex = " << processorInfoPtr->LastRowIndex()
                  << ", rowsCount = " << processorInfoPtr->RowsCount() << std::endl
                  << "FirstRowWithBordersIndex = " << processorInfoPtr->FirstRowWithBordersIndex()
                  << ", LastRowWithBordersIndex = " << processorInfoPtr->LastRowWithBordersIndex()
                  << ", RowsCountWithBorders = " << processorInfoPtr->RowsCountWithBorders() << std::endl;
        std::cout << "Creating ConjugateGradientAlgo ..." << std::endl;
#endif
        auto optimizationAlgo = new ConjugateGradientAlgo(netModelPtr, diffEquationPtr, approximateOperationsPtr,
                                                          processorInfoPtr);
        auto uValuesApproximate = optimizationAlgo->Init();
        auto uValues = optimizationAlgo->CalculateU();
        #ifdef DEBUG_MAIN
                    std::cout << "uValues  = " << std::endl << uValues << std::endl;
                    std::cout << "p = " << std::endl << uValuesApproximate << std::endl;
        #endif

#ifdef DEBUG_MAIN
        std::cout << "Created ConjugateGradientAlgo." << std::endl;
#endif
        optimizationAlgo->Process(uValuesApproximate, uValues);

    //        //    std::cout << netModelPtr->xValue(0) << " " << netModelPtr->yValue(0) << std::endl;
    //        //    std::cout << netModelPtr->xValue(pointsCount) << " " << netModelPtr->yValue(pointsCount) << std::endl;

//        writeValues(groundValuesFilename, uValues);
//        writeValues(approximateValuesFilename, uValuesApproximate);
        std::cout.rdbuf(coutbuf); //reset to standard output again
        out.close();
    }
    catch (const std::exception& e)
    {
        std::cout << rank << " : " << e.what() << std::endl;
    }
}
