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

#define DEBUG_MAIN
#define Print

void WriteValues(const char* filename, const DoubleMatrix& values)
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
            if (j != values.colsCount() - 1)
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

int IsPower(int number)
// the function returns log_{2}(Number) if it is integer. If not it returns (-1).
{
    unsigned int M;
    int p;
    if (number <= 0)
    {
        return(-1);
    }
    M = number; p = 0;
    while(M % 2 == 0)
    {
        ++p;
        M = M >> 1;
    }
    if((M >> 1) != 0)
    {
        return(-1);
    }
    else
    {
        return(p);
    }
}

int main(int argc, char *argv[])
{        
    int rank, processorsCount;
    try
    {
        double startTime, finishTime, elapsedTime, globalError;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processorsCount);

        startTime = MPI_Wtime();

        if (argc <= 2)
        {
            std::cerr << "Incorrect number of input params.\n";
            MPI_Finalize();
            exit(1);
        }
        auto N0 = std::stoi(argv[1]) + 1;
        auto N1 = std::stoi(argv[2]) + 1;

        int power = IsPower(processorsCount);
        if (power < 0)
        {
            std::cerr << "Incorrect number of processors. The number of procs must be a power of 2.\n";
            MPI_Finalize();
            exit(1);
        }

        auto processorInfoPtr = CreateProcessorData(processorsCount, N0, N1, power);

#ifdef DEBUG_MAIN
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        char fileName[25];
        sprintf(fileName, "out/out_%dx%d_%d__rank%d.txt", N0 - 1, N1 - 1,
                processorInfoPtr->processorsCount, processorInfoPtr->rank);

        std::ofstream out(fileName);
        std::cout.rdbuf(out.rdbuf());
#endif

        auto netModel = NetModel(xMinBoundary, xMaxBoundary, yMinBoundary, yMaxBoundary, N0, N1);
        netModel.InitModel(processorInfoPtr->FirstRowIndex(), processorInfoPtr->LastRowIndex(),
                               processorInfoPtr->FirstColIndex(), processorInfoPtr->LastColIndex());

        auto diffEquation = DifferentialEquationModel();
        auto approximateOperations = ApproximateOperations(netModel, *processorInfoPtr);

        auto optimizationAlgo = ConjugateGradientAlgo(netModel, diffEquation,
                                                      approximateOperations, *processorInfoPtr);


#ifdef Print
        if(rank == 0)
        {
            printf("k0 = %d, k1 = %d, n0 = %d, n1 = %d\n", processorInfoPtr->k0, processorInfoPtr->k1, processorInfoPtr->n0, processorInfoPtr->n1);
            printf("The number of processes ProcNum = 2^%d. It is split into %d x %d processes.\n"
                   "The number of nodes N0 = %d, N1 = %d. Blocks B(i,j) have size:\n", power,
                   processorInfoPtr->dims[0], processorInfoPtr->dims[1], N0,N1);

            if((processorInfoPtr->k0 > 0) && (processorInfoPtr->k1 > 0))
                printf("1 -->\t %d x %d iff i = 0 .. %d, j = 0 .. %d;\n", processorInfoPtr->n0 + 1,
                       processorInfoPtr->n1 + 1, processorInfoPtr->k0 - 1, processorInfoPtr->k1 - 1);
            if(processorInfoPtr->k1 > 0)
                printf("2 -->\t %d x %d iff i = %d .. %d, j = 0 .. %d;\n", processorInfoPtr->n0, processorInfoPtr->n1 + 1,
                       processorInfoPtr->k0, processorInfoPtr->dims[0] - 1, processorInfoPtr->k1 - 1);
            if(processorInfoPtr->k0 > 0)
                printf("3 -->\t %d x %d iff i = 0 .. %d, j = %d .. %d;\n", processorInfoPtr->n0 + 1, processorInfoPtr->n1,
                       processorInfoPtr->k0 - 1, processorInfoPtr->k1, processorInfoPtr->dims[1] - 1);

            printf("-->\t %d x %d iff i = %d .. %d, j = %d .. %d.\n", processorInfoPtr->n0, processorInfoPtr->n1,
                   processorInfoPtr->k0, processorInfoPtr->dims[0] - 1, processorInfoPtr->k1, processorInfoPtr->dims[1] - 1);
        }
#endif

#ifdef DEBUG_MAIN
        std::cout << "My Rank in Grid_Comm is " << rank << ". My topological coords is (" <<
                  processorInfoPtr->iCartIndex << "," << processorInfoPtr->jCartIndex << "). Domain size is " <<
                  processorInfoPtr->n0 << "x" << processorInfoPtr->n1 << " nodes.\n" <<
                  "My neighbours: left = " << processorInfoPtr->left << ", right = " << processorInfoPtr->right <<
                  ", down = " << processorInfoPtr->down << ", up = " << processorInfoPtr->up << ".\n" <<
                  "My block info: startColIndex = " << processorInfoPtr->startColIndex <<
                  ", colsCount = " << processorInfoPtr->colsCountValue << ", startRowIndex = " <<
                  processorInfoPtr->startRowIndex << ", rowsCount = " << processorInfoPtr->rowsCountValue <<
                  "\n";
        std::cout << "Created ConjugateGradientAlgo." << std::endl;
#endif

        auto uValues = optimizationAlgo.CalculateU();
        auto uValuesApproximate = optimizationAlgo.Process(&globalError, *uValues);

#ifdef DEBUG_MAIN
        std::cout << "Process finished, global error = " << globalError << std::endl;
        char outFileName[35];
        sprintf(outFileName, "../output/true/u_%dx%d_%d__rank%d.txt", N0 - 1, N1 - 1,
                processorInfoPtr->processorsCount, processorInfoPtr->rank);
        WriteValues(outFileName, *uValues);
        char outFileNameF[35];
        sprintf(outFileNameF, "../output/finish/p_%dx%d_%d__rank%d.txt", N0 - 1, N1 - 1,
                processorInfoPtr->processorsCount, processorInfoPtr->rank);
        WriteValues(outFileNameF, *uValuesApproximate);
#endif
#ifdef DEBUG_MAIN
        std::cout.rdbuf(coutbuf); //reset to standard output again
        out.close();
#endif
        if (processorInfoPtr->IsMainProcessor())
        {
            finishTime = MPI_Wtime();
            elapsedTime = finishTime - startTime;
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
