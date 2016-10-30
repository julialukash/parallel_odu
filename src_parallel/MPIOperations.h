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


double getMaxValueFromAllProcessors(double localValue);
double getFractionValueFromAllProcessors(double numerator, double denominator);

void RenewMatrixBoundRows(DoubleMatrix& values, std::shared_ptr<ProcessorsData> processorData, std::shared_ptr<NetModel> netModel);

#endif // MPIOPERATIONS_H
