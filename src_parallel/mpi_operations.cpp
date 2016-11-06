#include "mpi_operations.h"

#define DEBUG_MODE
void sendMatrix(const DoubleMatrix& values, int receiverRank, int tag)
{
    int rowsCount = values.rowsCount();
    int colsCount = values.colsCount();
    MPI_Send(&rowsCount, 1, MPI_INT, receiverRank, tag, MPI_COMM_WORLD);
    MPI_Send(&colsCount, 1, MPI_INT, receiverRank, tag, MPI_COMM_WORLD);
    for (int i = 0; i < rowsCount; ++i)
    {
        MPI_Send(&(values[i]), colsCount, MPI_DOUBLE, receiverRank, tag, MPI_COMM_WORLD);
    }
}

std::shared_ptr<DoubleMatrix> GatherUApproximateValuesMatrix(const ProcessorsData& processorInfoPtr,
                                           const NetModel &netModelPtr,
                                           const DoubleMatrix& uValuesApproximate)
{
    auto globalUValues = std::make_shared<DoubleMatrix>(1,1);
    if (processorInfoPtr.IsMainProcessor())
    {
        globalUValues = std::make_shared<DoubleMatrix>(netModelPtr.yPointsCount, netModelPtr.xPointsCount);
    }
    int recvcounts[processorInfoPtr.processorsCount], displs[processorInfoPtr.processorsCount];
    for (auto i = 0; i < processorInfoPtr.processorsCount; ++i)
    {
        auto processorParameters = ProcessorsData::GetProcessorParameters(netModelPtr.yPointsCount, i, processorInfoPtr.processorsCount);
        recvcounts[i] = processorParameters.first * netModelPtr.xPointsCount;
        displs[i] = processorParameters.second * netModelPtr.xPointsCount;
    }
    MPI_Gatherv(&(uValuesApproximate(0, 0)), recvcounts[processorInfoPtr.rank], MPI_DOUBLE,
                &((*globalUValues)(0, 0)), recvcounts, displs, MPI_DOUBLE, processorInfoPtr.mainProcessorRank, MPI_COMM_WORLD);
    return globalUValues;
}


std::pair<int, int> GetProcessorCoordsByRank(int rank, MPI_Comm gridComm, int ndims=2)
{
    int Coords[2];
    MPI_Cart_coords(gridComm, rank, ndims, Coords);
    return std::make_pair(Coords[0], Coords[1]);
}

std::shared_ptr<DoubleMatrix> GatherUValuesMatrix(const ProcessorsData& processorInfoPtr,
                                           const NetModel &netModelPtr,
                                           const DoubleMatrix& uValues)
{
    auto globalUValues = std::make_shared<DoubleMatrix>(1,1);
//    if (processorInfoPtr.IsMainProcessor())
//    {
//        globalUValues = std::make_shared<DoubleMatrix>(netModelPtr.xPointsCount, netModelPtr.yPointsCount);
//    }
//    int recvcounts[processorInfoPtr.processorsCount], displs[processorInfoPtr.processorsCount];
//    for (auto i = 0; i < processorInfoPtr.processorsCount; ++i)
//    {
//        auto processorCoordinates = GetProcessorCoordsByRank(i, processorInfoPtr.gridComm);
//        auto rowProcessorParameters = ProcessorsData::GetProcessorRowsParameters(processorCoordinates.second);
//        auto colProcessorParameters = ProcessorsData::GetProcessorColsParameters(processorCoordinates.first);
//        recvcounts[i] = processorParameters.first * netModelPtr.xPointsCount;
//        displs[i] = processorParameters.second * netModelPtr.xPointsCount;
//    }
//    MPI_Gatherv(&(uValuesApproximate(0, 0)), recvcounts[processorInfoPtr.rank], MPI_DOUBLE,
//                &((*globalUValues)(0, 0)), recvcounts, displs, MPI_DOUBLE, processorInfoPtr.mainProcessorRank, MPI_COMM_WORLD);
    return globalUValues;
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
    std::cout << "getFractionValueFromAllProcessors num  = " << numerator << ", " << denominator
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
    int nextProcessorRank = processorData.IsLastProcessor() ? MPI_PROC_NULL : processorData.rank + 1;
    int previousProcessorRank = processorData.IsFirstProcessor() ? MPI_PROC_NULL : processorData.rank - 1;

#ifdef DEBUG_MODE
    std::cout << "nextProcessorRank = " << nextProcessorRank << ", previousProcessorRank = " << previousProcessorRank << std::endl;
#endif

    // send to next processor last "no border" line
    // receive from prev processor first "border" line
    MPI_Sendrecv(&(values[processorData.RowsCountWithBorders() - 2]), netModel.xPointsCount, MPI_DOUBLE, nextProcessorRank, UP,
                 &(values[0]), netModel.xPointsCount, MPI_DOUBLE, previousProcessorRank, UP,
                 MPI_COMM_WORLD, &status);
    // send to prev processor first "no border" line
    // receive from next processor last "border" line
    MPI_Sendrecv(&(values[1]), netModel.xPointsCount, MPI_DOUBLE, previousProcessorRank, DOWN,
                 &(values[processorData.RowsCountWithBorders() - 1]), netModel.xPointsCount, MPI_DOUBLE, nextProcessorRank, DOWN,
                 MPI_COMM_WORLD, &status);
}
