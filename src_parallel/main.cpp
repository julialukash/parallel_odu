#include <iostream>
#include <fstream>

#include "approximate_operations.h"
#include "conjugate_gradient_algo.h"
#include "differential_equation_model.h"
#include "interface.h"
#include "mpi_operations.h"
#include "processors_data.h"

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

    for (int i = 0; i < values.rowsCount(); ++i)
    {
        for (int j = 0; j < values.colsCount(); ++j)
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
        double beginTime, elapsedTime, globalError;
        beginTime = clock();

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processorsCount);

        auto groundValuesFilename = argv[1];
        auto approximateValuesFilename = argv[2];
        auto pointsCount = std::stoi(argv[3]);

        if (processorsCount <= 0)// || (processorsCount > (pointsCount + 1) / 2))
        {
            std::cerr << "Incorrect number of processors\n";
            exit(1);
        }
        std::cout << "rank = " << rank << std::endl;


        auto netModelPtr = std::shared_ptr<NetModel>(new NetModel(xMinBoundary, xMaxBoundary,
                                                                  yMinBoundary, yMaxBoundary,
                                                                  pointsCount, pointsCount));

        auto processorInfoPtr = std::shared_ptr<ProcessorsData>(new ProcessorsData(rank, processorsCount));      
        // init processors with their part of data
        auto processorParameters = ProcessorsData::GetProcessorParameters(netModelPtr->yPointsCount, processorInfoPtr->rank, processorInfoPtr->processorsCount);
        processorInfoPtr->rowsCountValue = processorParameters.first;
        processorInfoPtr->startRowIndex = processorParameters.second;

        auto diffEquationPtr = std::shared_ptr<DifferentialEquationModel>(new DifferentialEquationModel());
        auto approximateOperationsPtr = std::shared_ptr<ApproximateOperations>(
                    new ApproximateOperations(*netModelPtr, *processorInfoPtr));

#ifdef DEBUG_MAIN
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        auto fileName = "out/out_rank" + std::to_string(processorInfoPtr->rank)  + ".txt";
        std::ofstream out(fileName);
        std::cout.rdbuf(out.rdbuf());
#endif
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
        auto optimizationAlgoPtr = std::shared_ptr<ConjugateGradientAlgo>(new ConjugateGradientAlgo(*netModelPtr, *diffEquationPtr, *approximateOperationsPtr,
                                                          *processorInfoPtr));
        auto uValuesApproximate = optimizationAlgoPtr->Init();
        auto uValues = optimizationAlgoPtr->CalculateU();
#ifdef DEBUG_MAIN
        std::cout << "main uValues  = " << std::endl << *uValues << std::endl;
        std::cout << "main p = " << std::endl << *uValuesApproximate << std::endl;
#endif

#ifdef DEBUG_MAIN
        std::cout << "Created ConjugateGradientAlgo." << std::endl;
#endif
        double localError = optimizationAlgoPtr->Process(uValuesApproximate, *uValues);
        globalError = GetMaxValueFromAllProcessors(localError);

#ifdef DEBUG_MAIN
        std::cout << "Process finished, error = " << localError << ", global = "
                  << globalError << ", u!!! = \n" << *uValuesApproximate << std::endl;
#endif
        // gather values
        auto globalUValues = GatherUApproximateValuesMatrix(*processorInfoPtr, *netModelPtr, *uValuesApproximate);
        if (processorInfoPtr->IsMainProcessor())
        {
#ifdef DEBUG_MAIN
            std::cout << "globalUValues = \n" << *globalUValues << std::endl;
#endif
            elapsedTime = double(clock() - beginTime) / CLOCKS_PER_SEC;
            std::cout << "Elapsed time: " <<  elapsedTime  << " sec." << std::endl
                      << "globalError: " << globalError << std::endl;
            writeValues(approximateValuesFilename, *globalUValues);
        }
#ifdef DEBUG_MAIN
        std::cout.rdbuf(coutbuf); //reset to standard output again
        out.close();
#endif
        if (processorInfoPtr->IsMainProcessor())
        {
            std::cout << "Elapsed time: " <<  elapsedTime  << " sec." << std::endl
                      << "globalError: " << globalError << std::endl;
        }
        MPI_Finalize();
    }
    catch (const std::exception& e)
    {
        std::cout << rank << " : " << e.what() << std::endl;
    }
}
