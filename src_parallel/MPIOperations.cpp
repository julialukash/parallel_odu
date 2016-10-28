#include "MPIOperations.h"
#include <mpi.h>



void sendMatrix(const DoubleMatrix& values, int receiverRank, int tag)
{
    int rowsCount = values.rowsCount();
    int colsCount = values.colsCount();
    MPI_Send(&rowsCount, 1, MPI_INT, receiverRank, tag, MPI_COMM_WORLD);
    MPI_Send(&colsCount, 1, MPI_INT, receiverRank, tag, MPI_COMM_WORLD);

    MPI_Send(&(values.matrix[0][0]), rowsCount * colsCount, MPI_DOUBLE, receiverRank, tag, MPI_COMM_WORLD);
}

std::shared_ptr<DoubleMatrix> receiveMatrix(int senderRank, int tag)
{
    MPI_Status status;
    int rowsCount;
    int colsCount;
    MPI_Recv(&rowsCount, 1, MPI_INT, senderRank, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&colsCount, 1, MPI_INT, senderRank, tag, MPI_COMM_WORLD, &status);

    auto values = std::shared_ptr<DoubleMatrix>(new DoubleMatrix(rowsCount, colsCount));
    MPI_Recv(&(values->matrix[0][0]), rowsCount * colsCount, MPI_DOUBLE, senderRank, tag, MPI_COMM_WORLD, &status);
}

void sendVector(const double *values, int size, int receiverRank, int tag)
{
//    MPI_Send(&size, 1, MPI_INT, receiverRank, tag, MPI_COMM_WORLD);
//    MPI_Send(&values[0], size, MPI_DOUBLE, receiverRank, tag, MPI_COMM_WORLD);
}

void receiveVector(double *values, int senderRank, int tag) {
    /*MPI_Status status;
    int size;
    MPI_Recv(&size, 1, MPI_INT, senderRank, tag, MPI_COMM_WORLD, &status);
    values = new double(size);
    MPI_Recv(&(*values)[0], size, MPI_DOUBLE, senderRank, tag, MPI_COMM_WORLD, &status)*/;
}

void sendValue(double value, int receiverRank, int tag)
{
    MPI_Send(&value, 1, MPI_DOUBLE, receiverRank, tag, MPI_COMM_WORLD);
}

void receiveValue(double* value, int senderRank, int tag)
{
    MPI_Status status;
    MPI_Recv(value, 1, MPI_DOUBLE, senderRank, tag, MPI_COMM_WORLD, &status);
}

void sendFlag(FlagType flag, int receiverRank, int tag)
{
    MPI_Send(&flag, 1, MPI_INT, receiverRank, tag, MPI_COMM_WORLD);
}

void receiveFlag(FlagType* flag, int senderRank, int tag) {
    MPI_Status status;
    MPI_Recv(flag, 1, MPI_DOUBLE, senderRank, tag, MPI_COMM_WORLD, &status);
}

double collectValueFromAll(int processorsCount)
{
    double value = 0;
    for (size_t i = 1; i < processorsCount; ++i) {
        double valuePart;
        receiveValue(&valuePart, i, i);
        value += valuePart;
    }
    return value;
}

void sendValueToAll(int processorsCount, double value)
{
    for (size_t i = 1; i < processorsCount; ++i)
    {
        sendValue(value, i, i);
    }
}

void sendFlagToAll(int processorsCount, FlagType flag)
{
    for (auto i = 1; i < processorsCount; ++i)
    {
        sendFlag(flag, i, i);
    }
}
