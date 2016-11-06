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
    APPROXIMATE_MATRIX,
    GROUND_MATRIX
};


std::shared_ptr<DoubleMatrix> GatherUApproximateValuesMatrix(const ProcessorsData& processorInfoPtr,
                                           const NetModel& netModelPtr,
                                           const DoubleMatrix& uValuesApproximate);

std::shared_ptr<DoubleMatrix> GatherUValuesMatrix(const ProcessorsData& processorInfoPtr,
                                           const NetModel &netModelPtr,
                                           const DoubleMatrix& uValues);

double GetMaxValueFromAllProcessors(double localValue);
double GetFractionValueFromAllProcessors(double numerator, double denominator);

void RenewMatrixBoundRows(DoubleMatrix& values, const ProcessorsData& processorData, const NetModel& netModel);

#endif // MPIOPERATIONS_H
