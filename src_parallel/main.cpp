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

#define DEBUG_MAIN
#define Print

void writeValues(const char* filename, const DoubleMatrix& values)
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
        auto N0 = std::stoi(argv[3]) + 1;
        auto N1 = std::stoi(argv[4]) + 1;

        MPI_Comm gridComm;             // this is a handler of a new communicator.
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

#ifdef DEBUG_MAIN
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        auto fileName = "out/out_rank" + std::to_string(rank)  + ".txt";
        std::ofstream out(fileName);
        std::cout.rdbuf(out.rdbuf());
#endif

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
        MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, true, &gridComm);
        MPI_Comm_rank(gridComm, &rank);
        MPI_Cart_coords(gridComm, rank, ndims, Coords);

        MPI_Cart_shift(gridComm, 0, 1, &left, &right);
        MPI_Cart_shift(gridComm, 1, 1, &down, &up);

        // setup row and col com
        MPI_Comm colComm, rowComm;
        int remainDims[2];
        remainDims[0] = 1; remainDims[1] = 0;
        MPI_Cart_sub(gridComm, remainDims, &rowComm);
        remainDims[0] = 0; remainDims[1] = 1;
        MPI_Cart_sub(gridComm, remainDims, &colComm);

        auto processorInfoPtr = std::shared_ptr<ProcessorsData>(new ProcessorsData(rank, processorsCount,
                                                                                   left, right,
                                                                                   down, up));

        // init processors with their part of data
        processorInfoPtr->InitComms(gridComm, rowComm, colComm);
        processorInfoPtr->InitCartParameters(n0, k0, n1, k1, N0, N1);
        processorInfoPtr->InitCartCoordinates(Coords[0], Coords[1]);
        auto processorParameters = ProcessorsData::GetProcessorRowsParameters(processorInfoPtr->N1, processorInfoPtr->n1, processorInfoPtr->k1, processorInfoPtr->jCartIndex);
        processorInfoPtr->InitRowsParameters(processorParameters);
        processorParameters = ProcessorsData::GetProcessorColsParameters(processorInfoPtr->n0, processorInfoPtr->k0, processorInfoPtr->iCartIndex);
        processorInfoPtr->InitColsParameters(processorParameters);

        auto netModelPtr = std::shared_ptr<NetModel>(new NetModel(xMinBoundary, xMaxBoundary,
                                                                  yMinBoundary, yMaxBoundary,
                                                                  N0, N1));
        netModelPtr->InitModel(processorInfoPtr->FirstRowIndex(), processorInfoPtr->LastRowIndex(),
                               processorInfoPtr->FirstColIndex(), processorInfoPtr->LastColIndex());

#ifdef DEBUG_MAIN
        std::cout << "XS = \n";
        for (int i = 0; i < netModelPtr->xValues.size(); ++i)
        {
            std::cout << netModelPtr->xValues[i] << " ";
        }
        std::cout << "\nYS = \n";
        for (int i = 0; i < netModelPtr->yValues.size(); ++i)
        {
            std::cout << netModelPtr->yValues[i] << " ";
        }
        std::cout << std::endl;
#endif
        auto diffEquationPtr = std::shared_ptr<DifferentialEquationModel>(new DifferentialEquationModel());
        auto approximateOperationsPtr = std::shared_ptr<ApproximateOperations>(
                    new ApproximateOperations(*netModelPtr, *processorInfoPtr));

#ifdef DEBUG_MAIN
        std::cout << "Finished" << std::endl;
        std::cout << "rank = " << processorInfoPtr->rank << ", processorsCount = " << processorInfoPtr->processorsCount << std::endl
                  << "FirstRowIndex = " << processorInfoPtr->FirstRowIndex()
                  << ", LastRowIndex = " << processorInfoPtr->LastRowIndex()
                  << ", rowsCount = " << processorInfoPtr->RowsCount() << std::endl
//                  << "FirstRowWithBordersIndex = " << processorInfoPtr->FirstRowWithBordersIndex()
//                  << ", LastRowWithBordersIndex = " << processorInfoPtr->LastRowWithBordersIndex()
                  << ", RowsCountWithBorders = " << processorInfoPtr->RowsCountWithBorders() << std::endl;
        std::cout << "Creating ConjugateGradientAlgo ..." << std::endl;
#endif
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
#ifdef DEBUG_MAIN
        std::cout << "My Rank in Grid_Comm is " << rank << ". My topological coords is (" <<
                  Coords[0] << "," << Coords[1] << "). Domain size is " <<
                  n0 << "x" << n1 << " nodes.\n" <<
                  "My neighbours: left = " << left << ", right = " << right <<
                  ", down = " << down << ", up = " << up << ".\n" <<
                  "My block info: startColIndex = " << processorInfoPtr->startColIndex <<
                  ", colsCount = " << processorInfoPtr->colsCountValue << ", startRowIndex = " <<
                  processorInfoPtr->startRowIndex << ", rowsCount = " << processorInfoPtr->rowsCountValue <<
                  "\n";
        std::cout << "Created ConjugateGradientAlgo." << std::endl;
#endif

        auto uValuesApproximate = optimizationAlgoPtr->Init();
        auto uValues = optimizationAlgoPtr->CalculateU();
#ifdef DEBUG_MAIN
        std::cout << "main uValues  = " << std::endl << *uValues << std::endl;
        std::cout << "main p = " << std::endl << *uValuesApproximate << std::endl;
        auto outFileName = "../output/init/init_rank" + std::to_string(rank)  + ".txt";
        writeValues(outFileName.c_str(), *uValuesApproximate);
        outFileName = "../output/true/u_rank" + std::to_string(rank)  + ".txt";
        writeValues(outFileName.c_str(), *uValues);
#endif
        double localError = optimizationAlgoPtr->Process(uValuesApproximate, *uValues);
        globalError = GetMaxValueFromAllProcessors(localError);

#ifdef DEBUG_MAIN
        std::cout << "Process finished, error = " << localError << ", global = "
                  << globalError << ", u!!! = \n" << *uValuesApproximate << std::endl;
#endif
//        // gather values
//        auto globalUValues = GatherUApproximateValuesMatrix(*processorInfoPtr, *netModelPtr, *uValuesApproximate);
//        if (processorInfoPtr->IsMainProcessor())
//        {
//#ifdef DEBUG_MAIN
//            std::cout << "globalUValues = \n" << *globalUValues << std::endl;
//#endif
//            elapsedTime = double(clock() - beginTime) / CLOCKS_PER_SEC;
//            std::cout << "Elapsed time: " <<  elapsedTime  << " sec." << std::endl
//                      << "globalError: " << globalError << std::endl;
//            writeValues(approximateValuesFilename, *globalUValues);
//        }
#ifdef DEBUG_MAIN
        std::cout.rdbuf(coutbuf); //reset to standard output again
        out.close();
#endif
//        if (processorInfoPtr->IsMainProcessor())
//        {
//            std::cout << "Elapsed time: " <<  elapsedTime  << " sec." << std::endl
//                      << "globalError: " << globalError << std::endl;
//        }
        MPI_Finalize();
    }
    catch (const std::exception& e)
    {
        std::cout << rank << " : " << e.what() << std::endl;
    }
}
