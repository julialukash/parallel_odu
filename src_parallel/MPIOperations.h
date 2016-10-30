#pragma once
#ifndef MPIOPERATIONS_H
#define MPIOPERATIONS_H

#include "Interface.h"
#include "ProcessorsData.h"

enum MessageTag
{
    UP,
    DOWN,
    APPROXIMATE_MATRIX = 77,
    GROUND_MATRIX = 78
};


std::shared_ptr<DoubleMatrix> GatherUApproximateValuesMatrix(const ProcessorsData& processorInfoPtr,
                                           const NetModel& netModelPtr,
                                           const DoubleMatrix& uValuesApproximate);

double GetMaxValueFromAllProcessors(double localValue);
double GetFractionValueFromAllProcessors(double numerator, double denominator);

void RenewMatrixBoundRows(DoubleMatrix& values, const ProcessorsData& processorData, const NetModel& netModel);

#endif // MPIOPERATIONS_H
