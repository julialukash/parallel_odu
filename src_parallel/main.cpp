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
            outputFile << values(i,j);
            if (j != values.colsCount())
            {
                outputFile << ", ";
            }
        }
        if (i != values.rowsCount() - 1)
        {
            outputFile << "\n";
        }
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

        auto processorInfoPtr = std::shared_ptr<ProcessorsData>(new ProcessorsData(rank, processorsCount));

        auto netModelPtr = std::shared_ptr<NetModel>(new NetModel(xMinBoundary, xMaxBoundary,
                                                                  yMinBoundary, yMaxBoundary,
                                                                  pointsCount, pointsCount));
        auto diffEquationPtr = std::shared_ptr<DifferentialEquationModel>(new DifferentialEquationModel());
        auto approximateOperationsPtr = std::shared_ptr<ApproximateOperations>(new ApproximateOperations(*netModelPtr));
        // init processors with their part of data
        auto processorParameters = ProcessorsData::GetProcessorParameters(netModelPtr->yPointsCount, processorInfoPtr->rank, processorInfoPtr->processorsCount);
        processorInfoPtr->rowsCountValue = processorParameters.first;
        processorInfoPtr->startRowIndex = processorParameters.second;
        auto optimizationAlgoPtr = std::shared_ptr<ConjugateGradientAlgo>(new ConjugateGradientAlgo(*netModelPtr, *diffEquationPtr, *approximateOperationsPtr,
                                                          *processorInfoPtr));
        auto uValuesApproximate = optimizationAlgoPtr->Init();
        auto uValues = optimizationAlgoPtr->CalculateU();
        double localError = optimizationAlgoPtr->Process(uValuesApproximate, *uValues);
        globalError = GetMaxValueFromAllProcessors(localError);
        // gather values
        auto globalUValues = GatherUApproximateValuesMatrix(*processorInfoPtr, *netModelPtr, *uValuesApproximate);
        if (processorInfoPtr->IsMainProcessor())
        {
            elapsedTime = double(clock() - beginTime) / CLOCKS_PER_SEC;
            std::cout << "Elapsed time: " <<  elapsedTime  << " sec." << std::endl
                      << "globalError: " << globalError << std::endl;
            writeValues(approximateValuesFilename, *globalUValues);
        }
        MPI_Finalize();
    }
    catch (const std::exception& e)
    {
        std::cout << rank << " : " << e.what() << std::endl;
    }
}
