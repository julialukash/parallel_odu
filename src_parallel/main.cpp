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

void writeValues(char* filename, std::vector<std::shared_ptr<DoubleMatrix> > globalValues)
{
    std::ofstream outputFile(filename);
    if (!outputFile.is_open())
    {
        std::cerr << "Incorrect output file " << filename;
        exit(1);
    }

    for (auto k = 0; k < globalValues.size(); ++k)
    {
        for (auto i = 0; i < globalValues[k]->rowsCount(); ++i)
        {
            for (auto j = 0; j < globalValues[k]->colsCount(); ++j)
            {
                outputFile << globalValues[k]->matrix[i][j] << " ";
            }
            outputFile << "\n";
        }
    }

    outputFile.close();
}

std::tuple<int, int> GetProcessorParameters(int pointsCount, int rank, int processorsCount)
{
    int rowsCount, firstRowIndex;
    rowsCount = pointsCount / processorsCount;
    auto leftRowsCount = pointsCount - rowsCount * processorsCount;
    if (rank < leftRowsCount)
    {
        rowsCount = rowsCount + 1;
        firstRowIndex = rank * rowsCount;
    }
    else
    {
        firstRowIndex = leftRowsCount * (rowsCount + 1) + (rank - leftRowsCount) * rowsCount;
    }
    std::cout << "left rows " << leftRowsCount << ", rc = " << rowsCount << ", fri = " << firstRowIndex << std::endl;
    return std::make_tuple(rowsCount, firstRowIndex);
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

        auto groundValuesFilename = argv[1];
        auto approximateValuesFilename = argv[2];
        auto pointsCount = std::stoi(argv[3]);

        auto netModelPtr = std::make_shared<NetModel>(xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary, pointsCount, pointsCount);
        auto diffEquationPtr = std::make_shared<DifferentialEquationModel>();
        auto approximateOperationsPtr = std::make_shared<ApproximateOperations>(netModelPtr);

#ifdef DEBUG_MAIN
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        auto fileName = "out/out_rank" + std::to_string(processorInfoPtr->rank)  + ".txt";
        std::ofstream out(fileName);
        std::cout.rdbuf(out.rdbuf());
#endif
        // init processors with their part of data
        auto processorParameters = GetProcessorParameters(netModelPtr->yPointsCount, processorInfoPtr->rank, processorInfoPtr->processorsCount);
        processorInfoPtr->rowsCountValue = std::get<0>(processorParameters);
        processorInfoPtr->startRowIndex = std::get<1>(processorParameters);
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
        double localError, globalError;
        localError = optimizationAlgo->Process(uValuesApproximate, uValues);
        globalError = getMaxValueFromAllProcessors(localError);

#ifdef DEBUG_MAIN
        std::cout << "Process finished, error = " << localError << ", global = "
                  << globalError << ", u = \n" << uValuesApproximate << std::endl;
#endif
        if (processorInfoPtr->IsMainProcessor())
        {
            std::vector<std::shared_ptr<DoubleMatrix>> globalUValues, globalUValuesApproximate;
            globalUValues.push_back(std::make_shared<DoubleMatrix>(uValues));
            globalUValuesApproximate.push_back(std::make_shared<DoubleMatrix>(uValuesApproximate));
#ifdef DEBUG_MAIN
            std::cout << "Gathering results..." << std::endl;
#endif
            for (int i = 1; i < processorInfoPtr->processorsCount; ++i)
            {
                auto localUValuesApproximate = receiveMatrix(i, APPROXIMATE_MATRIX);
//                auto localUValues = receiveMatrix(i, GROUND_MATRIX);
//#ifdef DEBUG_MAIN
//            std::cout << "receiveMatrix, i = "<< i << " tmp = " << localUValuesApproximate << "\n "
//                      << localUValuesApproximate->colsCount() << " " << localUValuesApproximate->rowsCount()
//                      << "\n" << localUValuesApproximate->matrix << std::endl;
//#endif
//                globalUValues.push_back(localUValues);
                globalUValuesApproximate.push_back(localUValuesApproximate);
            }
#ifdef DEBUG_MAIN
            std::cout << "Gathering results finished, globalUValuesCount = "<< globalUValuesApproximate.size() << std::endl;
#endif
//            writeValues(groundValuesFilename, globalUValues);
            writeValues(approximateValuesFilename, globalUValuesApproximate);
#ifdef DEBUG_MAIN
            std::cout << "writeValues finished." << std::endl;
#endif
        }
        else
        {
            sendMatrix(uValuesApproximate, processorInfoPtr->mainProcessorRank, APPROXIMATE_MATRIX);
//            sendMatrix(uValues, processorInfoPtr->mainProcessorRank, GROUND_MATRIX);
        }

//        double *globalUValues;
//        if (processorInfoPtr->IsMainProcessor())
//        {

//            globalUValues[netModelPtr->xPointsCount * netModelPtr->yPointsCount];/*
//            std::cout << "2*plainU = " << plainU[0] << std::endl;*/
//        }
//        int recvcounts[processorInfoPtr->processorsCount], displs[processorInfoPtr->processorsCount];
//        for (auto i = 0; i < processorInfoPtr->processorsCount; ++i)
//        {
//            auto processorParameters = GetProcessorParameters(netModelPtr->yPointsCount, i, processorInfoPtr->processorsCount);
//            recvcounts[i] = std::get<0>(processorParameters);
//            displs[i] = std::get<1>(processorParameters);
//        }

//        auto plainU = uValuesApproximate.PlainArray();
//        std::cout << "*plainU = " << plainU[0] << std::endl;
//        std::cout << "**plainU = " << plainU[0] << std::endl;
//#ifdef DEBUG_MAIN
//        std::cout << "recvcounts = \n" << recvcounts[0] << ", \ndispls = \n" << displs[0] << std::endl;
//        std::cout << "plainU = " << plainU[0] << std::endl;
//#endif

//        MPI_Gatherv(plainU, uValuesApproximate.colsCount() * uValuesApproximate.rowsCount(), MPI_DOUBLE,
//                    globalUValues, recvcounts, displs, MPI_DOUBLE, processorInfoPtr->mainProcessorRank, MPI_COMM_WORLD);

//        std::cout << "3plainU = " << plainU[0] << std::endl;

//        if (processorInfoPtr->IsMainProcessor())
//        {
//            std::cout << "4plainU = " << plainU[0] << std::endl;

//            auto tmp = DoubleMatrix(globalUValues, netModelPtr->yPointsCount, netModelPtr->xPointsCount);
//#ifdef DEBUG_MAIN
//            std::cout << "tmp = \n" << tmp << std::endl;
//#endif
//            writeValues(approximateValuesFilename, tmp);
//        }
#ifdef DEBUG_MAIN
        std::cout.rdbuf(coutbuf); //reset to standard output again
        out.close();
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << rank << " : " << e.what() << std::endl;
    }
}
