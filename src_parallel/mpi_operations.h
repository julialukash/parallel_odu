#pragma once
#ifndef MPIOPERATIONS_H
#define MPIOPERATIONS_H

#include "interface.h"
#include "processors_data.h"

#include <mpi.h>

enum MessageTag
{
    UP,
    DOWN,
    LEFT,
    RIGHT,
    APPROXIMATE_MATRIX,
    GROUND_MATRIX
};


double GetMaxValueFromAllProcessors(double localValue);
double GetFractionValueFromAllProcessors(double numerator, double denominator);

void RenewMatrixBoundRows(DoubleMatrix& values, const ProcessorsData& processorData, const NetModel& netModel);
void RenewMatrixBoundCols(DoubleMatrix& values, const ProcessorsData& processorData, const NetModel& netModel);

std::shared_ptr<ProcessorsData> CreateProcessorData(int processorsCount, int N0, int N1, int power);

#endif // MPIOPERATIONS_H
