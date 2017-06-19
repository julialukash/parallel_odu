#ifndef DERIVATOR_H
#define DERIVATOR_H

#include <algorithm>

#include "interface.h"
#include "processors_data.h"


class ApproximateOperations
{
private:
    const NetModel& netModel;
    const ProcessorsData& processorData;
public:
    ApproximateOperations(const NetModel& model, const ProcessorsData& processorInfo);
    // note: calculates -laplass(currentValues)
    void CalculateLaplassCUDA(double *devCurrentValues, double *devLaplassValues, double* devXValues, double* devYValues) const;
    double ScalarProductCUDA(double* devCurrentValues, double* devOtherValues, double* devXValues, double* devYValues, double *devResultMatrix, double *devAuxiliaryMatrix) const;
    double MaxNormValueCUDA(double *devMatrix, int numObjects, double *devResultMatrix) const;

};

#endif // DERIVATOR_H
