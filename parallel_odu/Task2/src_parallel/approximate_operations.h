#ifndef DERIVATOR_H
#define DERIVATOR_H

#include <algorithm>

#include "interface.h"
#include "processors_data.h"

#define PARALLEL_O

class ApproximateOperations
{
private:
    const NetModel& netModel;
    const ProcessorsData& processorData;
public:
    ApproximateOperations(const NetModel& model, const ProcessorsData& processorData):
        netModel(model), processorData(processorData)
    {
    }

    // note: calculates -laplass(currentValues)
    DoubleMatrix* CalculateLaplass(const DoubleMatrix& currentValues) const
    {
        DoubleMatrix* laplassValues = new DoubleMatrix(processorData.RowsCountWithBorders(), processorData.ColsCountWithBorders());
#ifdef PARALLEL_O
        #pragma omp parallel for
#endif
        for (int i = processorData.FirstInnerRowRelativeIndex(); i <= processorData.LastInnerRowRelativeIndex(); ++i)
        {
            for (int j = processorData.FirstInnerColRelativeIndex(); j <= processorData.LastInnerColRelativeIndex(); ++j)
            {
                double xPart = (currentValues(i, j) - currentValues(i, j - 1)) / netModel.xStep(j - 1) -
                             (currentValues(i, j + 1) - currentValues(i, j)) / netModel.xStep(j);
                double yPart = (currentValues(i, j) - currentValues(i - 1, j)) / netModel.yStep(i - 1) -
                             (currentValues(i + 1, j) - currentValues(i, j)) / netModel.yStep(i);
                (*laplassValues)(i, j) = xPart / netModel.xAverageStep(j) + yPart / netModel.yAverageStep(i);
            }
        }
        return laplassValues;
    }

    double ScalarProduct(const DoubleMatrix& currentValues, const DoubleMatrix& otherValues) const
    {
        double prodValue = 0;
#ifdef PARALLEL_O
        #pragma omp parallel for reduction(+:prodValue)
#endif
        for (int i = processorData.FirstInnerRowRelativeIndex(); i <= processorData.LastInnerRowRelativeIndex(); ++i)
        {
            for (int j = processorData.FirstInnerColRelativeIndex(); j <= processorData.LastInnerColRelativeIndex(); ++j)
            {
                prodValue += netModel.xAverageStep(j) * netModel.yAverageStep(i) *
                                      currentValues(i, j) * otherValues(i, j);
            }
        }
        return prodValue;
    }


    double MaxNormValue(const DoubleMatrix& currentValues) const
    {
        double maxNorm = -1;
#ifdef PARALLEL_O
        #pragma omp parallel
#endif
        {
            double localMax = -1;
#ifdef PARALLEL_O
            #pragma omp for nowait
#endif
            for (int i = 1; i < currentValues.rowsCount() - 1; ++i)
            {
                for (int j = 1; j < currentValues.colsCount() - 1; ++j)
                {
                    double absValue = fabs(currentValues(i, j));
                    if (absValue > localMax)
                    {
                        localMax = absValue;
                    }
                }
            }
#ifdef PARALLEL_O
            #pragma omp critical
#endif
            {
                maxNorm = std::max(maxNorm, localMax);
            }
        }
        return maxNorm;
    }
};

#endif // DERIVATOR_H
