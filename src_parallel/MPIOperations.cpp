#include "MPIOperations.h"
#include <mpi.h>

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

std::shared_ptr<DoubleMatrix> receiveMatrix(int senderRank, int tag)
{
    MPI_Status status;
    int rowsCount;
    int colsCount;
    MPI_Recv(&rowsCount, 1, MPI_INT, senderRank, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&colsCount, 1, MPI_INT, senderRank, tag, MPI_COMM_WORLD, &status);


    auto values = std::shared_ptr<DoubleMatrix>(new DoubleMatrix(rowsCount, colsCount));
    for (int i = 0; i < rowsCount; ++i)
    {
        MPI_Recv(&(values->matrix[i]), colsCount, MPI_DOUBLE, senderRank, tag, MPI_COMM_WORLD, &status);
    }
    return values;
}

double getMaxValueFromAllProcessors(double localValue)
{
    double globalValue;
    MPI_Allreduce(&localValue, &globalValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return globalValue;
}

double getFractionValueFromAllProcessors(double numerator, double denominator)
{
    double localValue[2] = {numerator, denominator};
    double globalValue[2] = {0, 0};
    MPI_Allreduce(localValue, globalValue, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//#ifdef DEBUG_MODE
    std::cout << "getFractionValueFromAllProcessors num  = " << numerator << ", " << denominator
              << ", local = " << *localValue  << " " << *(localValue + 1)
              << ", global = " << *globalValue << " " <<  *(globalValue + 1) << std::endl;
//              << "tau = " << globalValue[0] / globalValue[1] << std::endl;
//#endif
    return globalValue[0] / globalValue[1];
}

void RenewMatrixBoundRows(DoubleMatrix& values, std::shared_ptr<ProcessorsData> processorData, std::shared_ptr<NetModel> netModel)
{
#ifdef DEBUG_MODE
    std::cout << "RenewBoundRows \n" << values << std::endl;
#endif
    MPI_Status status;
    int nextProcessorRank = processorData->IsLastProcessor() ? MPI_PROC_NULL : processorData->rank + 1;
    int previousProcessorRank = processorData->IsFirstProcessor() ? MPI_PROC_NULL : processorData->rank - 1;

#ifdef DEBUG_MODE
    std::cout << "nextProcessorRank = " << nextProcessorRank << ", previousProcessorRank = " << previousProcessorRank << std::endl;
#endif

    // send to next processor last "no border" line
    // receive from prev processor first "border" line
    MPI_Sendrecv(&(values[processorData->RowsCountWithBorders() - 2]), netModel->xPointsCount, MPI_DOUBLE, nextProcessorRank, UP,
                 &(values[0]), netModel->xPointsCount, MPI_DOUBLE, previousProcessorRank, UP,
                 MPI_COMM_WORLD, &status);
    // send to prev processor first "no border" line
    // receive from next processor last "border" line
    MPI_Sendrecv(&(values[1]), netModel->xPointsCount, MPI_DOUBLE, previousProcessorRank, DOWN,
                 &(values[processorData->RowsCountWithBorders() - 1]), netModel->xPointsCount, MPI_DOUBLE, nextProcessorRank, DOWN,
                 MPI_COMM_WORLD, &status);
}
