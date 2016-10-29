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
        MPI_Send(&(values.matrix[i][0]), colsCount, MPI_DOUBLE, receiverRank, tag, MPI_COMM_WORLD);
    }
}

std::shared_ptr<DoubleMatrix> receiveMatrix(int senderRank, int tag)
{
//    std::cout << "receive " << std::endl;
    MPI_Status status;
    int rowsCount;
    int colsCount;
    MPI_Recv(&rowsCount, 1, MPI_INT, senderRank, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&colsCount, 1, MPI_INT, senderRank, tag, MPI_COMM_WORLD, &status);


    auto values = std::shared_ptr<DoubleMatrix>(new DoubleMatrix(rowsCount, colsCount));
    for (int i = 0; i < rowsCount; ++i)
    {
        MPI_Recv(&(values->matrix[i][0]), colsCount, MPI_DOUBLE, senderRank, tag, MPI_COMM_WORLD, &status);
    }/*
    std::cout << "senderRank " << senderRank << " " << rowsCount << ", " << colsCount
              << "values = \n" << values->matrix << std::endl;*/

    return values;
}

