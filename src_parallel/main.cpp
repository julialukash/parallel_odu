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


void InitData()
{

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
        auto processorInfoPtr = std::make_shared<ProcessorsData>(rank, processorsCount);

        auto groundValuesFilename = argv[1];
        auto approximateValuesFilename = argv[2];
        auto pointsCount = std::stoi(argv[3]);

        auto netModelPtr = std::make_shared<NetModel>(xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary, pointsCount, pointsCount);
        auto diffEquationPtr = std::make_shared<DifferentialEquationModel>();
        auto approximateOperationsPtr = std::make_shared<ApproximateOperations>(netModelPtr);

        if (processorInfoPtr->IsMainProcessor())
        {
            auto iteration = 0;
//            while (true)
//            {
//                ++iteration;
                // start new iter
//                sendFlagToAll(processorInfoPtr->processorsCount, START_ITER);

//                // wait for parts of alpha, sum them and send back
//                double alpha_den = collect_value_from_all(processorInfoPtr->processorCount);
//                double alpha_nom = collect_value_from_all(num_processors);
//                double alpha = alpha_nom / alpha_den;
//                send_value_to_all(num_processors, alpha);

//                // wait for parts of tau, sum them and send back
//                double tau_den = collect_value_from_all(num_processors);
//                double tau_nom = collect_value_from_all(num_processors);
//                double tau = tau_nom / tau_den;
//                send_value_to_all(num_processors, tau);

//                // wait for the end of the iteration and collect errors
//                double not_finished = collect_value_from_all(num_processors);

//                if (not_finished < EPS) {
//                    sendFlagToAll(processorInfoPtr->processorsCount, TERMINATE);

//                    // receive results
//                    error = sqrt(collect_value_from_all(num_processors));
//                    std::vector<std::shared_ptr<DM> > all_values;
//                    for (int i = 1; i < num_processors; ++i) {
//                        all_values.push_back(receive_matrix(i, i));
//                    }
//                    print_results(all_values, grid, functions);
//                    break;
//                }
//            }

//            std::cout << processorInfoPtr->rank  << ": Finished! Elapsed time: "
//                      << float(clock() - beginTime) / CLOCKS_PER_SEC << " sec." << std::endl
//                      << "Num iters processed: " << iteration << std::endl;

        }
        else
        {
            // init processors with their part of data
            processorInfoPtr->rowsCountValue = (netModelPtr->yPointsCount) / (processorInfoPtr->processorsCount - 1);
            auto leftRowsCount = netModelPtr->yPointsCount - processorInfoPtr->rowsCountValue  * (processorInfoPtr->processorsCount - 1);

            processorInfoPtr->startRowIndex = (processorInfoPtr->rank - 1) * processorInfoPtr->rowsCountValue;
            if (leftRowsCount != 0 & processorInfoPtr->IsLastProcessor())
            {
                processorInfoPtr->rowsCountValue = processorInfoPtr->rowsCountValue + leftRowsCount;
            }
            std::cout << "rank = " << processorInfoPtr->rank << ", processorsCount = " << processorInfoPtr->processorsCount << std::endl
                      << "startRowIndex = " << processorInfoPtr->FirstRowIndex() << std::endl
                      << ", endRowIndex = " << processorInfoPtr->LastRowIndex() << std::endl
                      << ", rowsCount = " << processorInfoPtr->RowsCount() << std::endl
                      << "leftRowsCount = " << leftRowsCount << std::endl;
        }
//        auto optimizationAlgo = new ConjugateGradientAlgo(netModelPtr, diffEquationPtr, approximateOperationsPtr);

//        //    std::cout << netModelPtr->xValue(0) << " " << netModelPtr->yValue(0) << std::endl;
//        //    std::cout << netModelPtr->xValue(pointsCount) << " " << netModelPtr->yValue(pointsCount) << std::endl;

//        auto uValues = diffEquationPtr->CalculateUValues(netModelPtr);

//        auto begin = omp_get_wtime();

//        optimizationAlgo->Process(uValuesApproximate, uValues);

//        auto time_elapsed = omp_get_wtime() - begin;
//        std::cout << "Elapsed time is " << time_elapsed << " sec" << std::endl;

//        writeValues(groundValuesFilename, uValues);
//        writeValues(approximateValuesFilename, uValuesApproximate);
    }
    catch (const std::exception& e)
    {
        std::cout << rank << " : " << e.what() << std::endl;
    }
}
