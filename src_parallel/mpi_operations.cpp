#include "mpi_operations.h"

//#define DEBUG_MODE

const int ndims = 2;

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
    return p0;
}

std::shared_ptr<ProcessorsData> CreateProcessorData(int processorsCount, int N0, int N1, int power)
{
    MPI_Comm gridComm;             // this is a handler of a new communicator.
    int Coords[2];
    int periods[2] = {0,0};         // it is used for creating processes topology.
    int rank, left, right, up, down;

    int p0 = SplitFunction(N0, N1, power);
    int p1 = power - p0;

    auto processorInfoPtr = std::shared_ptr<ProcessorsData>(new ProcessorsData(processorsCount));
    processorInfoPtr->InitCartParameters(p0, p1, N0, N1);

    // the cartesian topology of processes is being created ...
    MPI_Cart_create(MPI_COMM_WORLD, ndims, processorInfoPtr->dims, periods, true, &gridComm);
    MPI_Comm_rank(gridComm, &rank);
    MPI_Cart_coords(gridComm, rank, ndims, Coords);

    MPI_Cart_shift(gridComm, 0, 1, &left, &right);
    MPI_Cart_shift(gridComm, 1, 1, &down, &up);

    processorInfoPtr->left = left;       processorInfoPtr->up = up;
    processorInfoPtr->right = right;     processorInfoPtr->down = down;

    // init processors with their part of data
    processorInfoPtr->rank = rank;
    processorInfoPtr->gridComm = gridComm;
    processorInfoPtr->iCartIndex = Coords[0];
    processorInfoPtr->jCartIndex = Coords[1];
    processorInfoPtr->InitProcessorRowsParameters();
    processorInfoPtr->InitProcessorColsParameters();
    return processorInfoPtr;
}

std::pair<int, int> GetProcessorCoordsByRank(int rank, MPI_Comm gridComm, int ndims=2)
{
    int Coords[2];
    MPI_Cart_coords(gridComm, rank, ndims, Coords);
    return std::make_pair(Coords[0], Coords[1]);
}

double GetMaxValueFromAllProcessors(double localValue)
{
    double globalValue;
    MPI_Allreduce(&localValue, &globalValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return globalValue;
}

double GetFractionValueFromAllProcessors(double numerator, double denominator)
{
    double localValue[2] = {numerator, denominator};
    double globalValue[2] = {0, 0};
    MPI_Allreduce(localValue, globalValue, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef DEBUG_MODE
    std::cout << "GetFractionValueFromAllProcessors num  = " << numerator << ", " << denominator
              << ", local = " << *localValue  << " " << *(localValue + 1)
              << ", global = " << *globalValue << " " <<  *(globalValue + 1) << std::endl;
//              << "tau = " << globalValue[0] / globalValue[1] << std::endl;
#endif
    return globalValue[0] / globalValue[1];
}

void RenewMatrixBoundRows(DoubleMatrix& values, const ProcessorsData& processorData, const NetModel& netModel)
{
#ifdef DEBUG_MODE
    std::cout << "RenewBoundRows \n" << values << std::endl;
#endif
    MPI_Status status;
    int downProcessorRank = processorData.IsLastProcessor() ? MPI_PROC_NULL : processorData.down;
    int upProcessorRank = processorData.IsFirstProcessor() ? MPI_PROC_NULL : processorData.up;

#ifdef DEBUG_MODE
    std::cout << "nextProcessorRank = " << downProcessorRank << ", previousProcessorRank = " << upProcessorRank << std::endl;
#endif

    // send to next processor last "no border" line
    // receive from prev processor first "border" line
    MPI_Sendrecv(&(values[processorData.LastOwnRowRelativeIndex()]), processorData.ColsCountWithBorders(), MPI_DOUBLE, downProcessorRank, UP,
                 &(values[0]), processorData.ColsCountWithBorders(), MPI_DOUBLE, upProcessorRank, UP,
                 processorData.gridComm, &status);
    // send to prev processor first "no border" line
    // receive from next processor last "border" line
    MPI_Sendrecv(&(values[processorData.FirstOwnRowRelativeIndex()]), processorData.ColsCountWithBorders(), MPI_DOUBLE, upProcessorRank, DOWN,
                 &(values[processorData.RowsCountWithBorders() - 1]), processorData.ColsCountWithBorders(), MPI_DOUBLE, downProcessorRank, DOWN,
                 processorData.gridComm, &status);
#ifdef DEBUG_MODE
    std::cout << "RenewBoundRows finished \n" << values << std::endl;
#endif
}

void RenewMatrixBoundCols(DoubleMatrix& values, const ProcessorsData& processorData, const NetModel& netModel)
{
#ifdef DEBUG_MODE
    std::cout << "RenewBoundCols \n" << values << std::endl;
#endif
    MPI_Status status;
    int leftProcessorRank = processorData.IsLeftProcessor() ? MPI_PROC_NULL : processorData.left;
    int rightProcessorRank = processorData.IsRightProcessor() ? MPI_PROC_NULL : processorData.right;

#ifdef DEBUG_MODE
    std::cout << "leftProcessorRank = " << leftProcessorRank << ", rightProcessorRank = " << rightProcessorRank << std::endl;
#endif
    // tmp vectors for keeping left and right cols
    auto leftOwn = values.CropMatrix(0, values.rowsCount(), processorData.FirstOwnColRelativeIndex(), 1);
    auto rightOwn = values.CropMatrix(0, values.rowsCount(), processorData.LastOwnColRelativeIndex(), 1);
    auto leftBorder = values.CropMatrix(0, values.rowsCount(), 0, 1);
    auto rightBorder = values.CropMatrix(0, values.rowsCount(), processorData.ColsCountWithBorders() - 1, 1);

#ifdef DEBUG_MODE
    std::cout << "leftOwn = \n" << *leftOwn << ", rightOwn = \n" << *rightOwn << std::endl;
    std::cout << "leftBorder = \n" << *leftBorder << ", rightBorder = \n" << *rightBorder << std::endl;
#endif

    // send to right processor last "no border" line
    // receive from left processor first "border" line
    MPI_Sendrecv(&((*rightOwn)[0]), leftOwn->rowsCount(), MPI_DOUBLE, rightProcessorRank, LEFT,
                 &((*leftBorder)[0]), leftOwn->rowsCount(), MPI_DOUBLE, leftProcessorRank, LEFT,
                 processorData.gridComm, &status);
#ifdef DEBUG_MODE
//    std::cout << "first change status = " << &status << std::endl;
#endif
    // send to left processor first "no border" line
    // receive from next processor last "border" line
    MPI_Sendrecv(&((*leftOwn)[0]), leftOwn->rowsCount(), MPI_DOUBLE, leftProcessorRank, RIGHT,
                 &((*rightBorder)[0]), leftOwn->rowsCount(), MPI_DOUBLE, rightProcessorRank, RIGHT,
                 processorData.gridComm, &status);
#ifdef DEBUG_MODE
//    std::cout << "second change status = " << &status << std::endl;
#endif
    values.SetNewColumn(*leftBorder, 0);
    values.SetNewColumn(*rightBorder, processorData.ColsCountWithBorders() - 1);
#ifdef DEBUG_MODE
    std::cout << "after change \n" << std::endl;
    std::cout << "leftOwn = \n" << *leftOwn << ", rightOwn = \n" << *rightOwn << std::endl;
    std::cout << "leftBorder = \n" << *leftBorder << ", rightBorder = \n" << *rightBorder << std::endl;
    std::cout << "RenewBoundCols finished \n" << values << std::endl;
#endif
}
