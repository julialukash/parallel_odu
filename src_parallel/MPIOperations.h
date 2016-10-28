#pragma once
#ifndef MPIOPERATIONS_H
#define MPIOPERATIONS_H

#include "Interface.h"


enum FlagType
{
  START_ITER,
  TERMINATE,
};

void sendMatrix(const DoubleMatrix& values, int receiverRank, int tag);
std::shared_ptr<DoubleMatrix> receiveMatrix(int senderRank, int tag);

void sendVector(const double* values, int size, int receiverRank, int tag);
void receiveVector(double* values, int senderRank, int tag);

void sendValue(double value, int receiverRank, int tag);
void receiveValue(double* value, int senderRank, int tag);

void sendFlag(FlagType flag, int receiverRank, int tag);
void receiveFlag(FlagType* flag, int senderRank, int tag);

double collectValueFromAll(int processorsCount);
void sendValueToAll(int processorsCount, double value);
void sendFlagToAll(int processorsCount, FlagType flag);

#endif // MPIOPERATIONS_H
