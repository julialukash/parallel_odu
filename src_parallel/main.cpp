#include <iostream>
#include <fstream>

#include "approximate_operations.h"
#include "conjugate_gradient_algo.h"
#include "differential_equation_model.h"
#include "interface.h"
#include "mpi_operations.h"
#include "processors_data.h"

const int ndims = 2;
const double xMinBoundary = 0;
const double xMaxBoundary = 2;
const double yMinBoundary = 0;
const double yMaxBoundary = 2;
const double eps = 1e-4;

#define Print


#define BLOCK_LOW(id,p,n)   ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n)  (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n)  (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)

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


int SplitFunction(int N0, int N1, int p)
// This is the splitting procedure of proc. number p. The integer p0
// is calculated such that abs(N0/p0 - N1/(p-p0)) --> min.
{
    float n0, n1;
    int p0, i;

    n0 = (float) N0; n1 = (float) N1;
    p0 = 0;

    for(i = 0; i < p; i++)
    {

        if(n0 > n1)
        {
            n0 = n0 / 2.0;
            ++p0;
        }
        else
        {
            n1 = n1 / 2.0;
        }
    }
    return(p0);
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

        if (argc <= 4)
        {
            std::cerr << "Incorrect number of input params.\n";
            MPI_Finalize();
            exit(1);
        }
        auto groundValuesFilename = argv[1];
        auto approximateValuesFilename = argv[2];
        auto N0 = std::stoi(argv[3]);
        auto N1 = std::stoi(argv[4]);

        MPI_Comm Grid_Comm;             // this is a handler of a new communicator.
        int dims[2], Coords[2];
        int periods[2] = {0,0};         // it is used for creating processes topology.
        int left, right, up, down;
        int power, p0, p1, n0, k0, n1, k1;
        if ((power = IsPower(processorsCount)) < 0)// || (processorsCount > (pointsCount + 1) / 2))
        {
            std::cerr << "Incorrect number of processors. The number of procs must be a power of 2.\n";
            MPI_Finalize();
            exit(1);
        }

        p0 = SplitFunction(N0, N1, power);
        p1 = power - p0;

        dims[0] = (unsigned int) 1 << p0;   dims[1] = (unsigned int) 1 << p1;
        n0 = N0 >> p0;                      n1 = N1 >> p1;
        k0 = N0 - dims[0]*n0;               k1 = N1 - dims[1]*n1;

#ifdef Print
        if(rank == 0)
        {
            printf("k0 = %d, k1 = %d, n0 = %d, n1 = %d\n", k0, k1, n0, n1);
            printf("The number of processes ProcNum = 2^%d. It is split into %d x %d processes.\n"
                   "The number of nodes N0 = %d, N1 = %d. Blocks B(i,j) have size:\n", power, dims[0],dims[1], N0,N1);

            if((k0 > 0)&&(k1 > 0))
                printf("1 -->\t %d x %d iff i = 0 .. %d, j = 0 .. %d;\n", n0+1,n1+1, k0-1,k1-1);
            if(k1 > 0)
                printf("2 -->\t %d x %d iff i = %d .. %d, j = 0 .. %d;\n", n0,n1+1, k0,dims[0]-1, k1-1);
            if(k0 > 0)
                printf("3 -->\t %d x %d iff i = 0 .. %d, j = %d .. %d;\n", n0+1,n1, k0-1, k1,dims[1]-1);

            printf("-->\t %d x %d iff i = %d .. %d, j = %d .. %d.\n", n0,n1, k0,dims[0]-1, k1,dims[1]-1);
        }
#endif

        // the cartesian topology of processes is being created ...
        MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, true, &Grid_Comm);
        MPI_Comm_rank(Grid_Comm, &rank);
        MPI_Cart_coords(Grid_Comm, rank, ndims, Coords);

        MPI_Cart_shift(Grid_Comm, 0, 1, &left, &right);
        MPI_Cart_shift(Grid_Comm, 1, 1, &down, &up);

        auto netModelPtr = std::shared_ptr<NetModel>(new NetModel(xMinBoundary, xMaxBoundary,
                                                                  yMinBoundary, yMaxBoundary,
                                                                  N0, N0));
        auto processorInfoPtr = std::shared_ptr<ProcessorsData>(new ProcessorsData(rank, processorsCount,
                                                                                   left, right,
                                                                                   down, up));

        auto diffEquationPtr = std::shared_ptr<DifferentialEquationModel>(new DifferentialEquationModel());
        auto approximateOperationsPtr = std::shared_ptr<ApproximateOperations>(new ApproximateOperations(*netModelPtr));
        // init processors with their part of data
        auto processorParameters = ProcessorsData::GetProcessorRowsParameters(n1, k1, Coords[1]);
        processorInfoPtr->rowsCountValue = processorParameters.first;
        processorInfoPtr->startRowIndex = processorParameters.second;
        processorParameters = ProcessorsData::GetProcessorColsParameters(n0, k0, Coords[0]);
        processorInfoPtr->colsCountValue = processorParameters.first;
        processorInfoPtr->startColIndex = processorParameters.second;
        auto optimizationAlgoPtr = std::shared_ptr<ConjugateGradientAlgo>(new ConjugateGradientAlgo(*netModelPtr, *diffEquationPtr, *approximateOperationsPtr,
                                                          *processorInfoPtr));

        if(Coords[0] < k0)
            ++n0;
        if(Coords[1] < k1)
            ++n1;

#ifdef Print
        printf("My Rank in Grid_Comm is %d. My topological coords is (%d,%d). Domain size is %d x %d nodes.\n"
               "My neighbours: left = %d, right = %d, down = %d, up = %d.\n"
               "My block info: startColIndex = %d, colsCount = %d, startRowIndex = %d, rowsCount = %d.\n",
               rank, Coords[0], Coords[1], n0, n1, left, right, down,up,
               processorInfoPtr->startColIndex, processorInfoPtr->colsCountValue,
               processorInfoPtr->startRowIndex, processorInfoPtr->rowsCountValue);
#endif


//        auto uValuesApproximate = optimizationAlgoPtr->Init();
//        auto uValues = optimizationAlgoPtr->CalculateU();
//        double localError = optimizationAlgoPtr->Process(uValuesApproximate, *uValues);
//        globalError = GetMaxValueFromAllProcessors(localError);
//        // gather values
//        auto globalUValues = GatherUApproximateValuesMatrix(*processorInfoPtr, *netModelPtr, *uValuesApproximate);
//        if (processorInfoPtr->IsMainProcessor())
//        {
//            elapsedTime = double(clock() - beginTime) / CLOCKS_PER_SEC;
//            std::cout << "Elapsed time: " <<  elapsedTime  << " sec." << std::endl
//                      << "globalError: " << globalError << std::endl;
//            writeValues(approximateValuesFilename, *globalUValues);
//        }
        MPI_Finalize();
    }
    catch (const std::exception& e)
    {
        std::cout << rank << " : " << e.what() << std::endl;
    }
}
