#ifndef DERIVATOR_H
#define DERIVATOR_H

#include <algorithm>

#include "interface.h"
#include "processors_data.h"


class ApproximateOperations
{
private:
    const NetModel& netModel;
public:
    ApproximateOperations(const NetModel& model): netModel(model)
    {
    }

    // note: calculates -laplass(currentValues)
    std::shared_ptr<DoubleMatrix> CalculateLaplass(const DoubleMatrix& currentValues, const ProcessorsData& processorData) const
    {        
#ifdef DEBUG_MODE
        std::cout <<"ApproximateOperations.CalculateLaplass currentValues = \n" << currentValues << std::endl;
        std::cout <<"ApproximateOperations.CalculateLaplass startIndex = " << startIndex << ", endIndex = " << endIndex << std::endl;
#endif
        auto laplassValues = std::make_shared<DoubleMatrix>(processorData.RowsCountWithBorders(), netModel.xPointsCount);
        for (int i = processorData.FirstRowRelativeIndex(); i <= processorData.LastRowRelativeIndex(); ++i)
        {
            int iNetIndex = i - processorData.FirstRowRelativeIndex() + processorData.FirstRowIndex();
            for (int j = 1; j < laplassValues->colsCount() - 1; ++j)
            {
                double xPart = (currentValues(i, j) - currentValues(i - 1, j)) / netModel.xStep(iNetIndex - 1) -
                             (currentValues(i + 1, j) - currentValues(i, j)) / netModel.xStep(iNetIndex);
                double yPart = (currentValues(i, j) - currentValues(i, j - 1)) / netModel.yStep(j - 1) -
                             (currentValues(i, j + 1) - currentValues(i, j)) / netModel.yStep(j);
                (*laplassValues)(i, j) = xPart / netModel.xAverageStep(iNetIndex) + yPart / netModel.yAverageStep(j);
            }
        }
        return laplassValues;
    }

    double ScalarProduct(const DoubleMatrix& currentValues, const DoubleMatrix& otherValues, const ProcessorsData& processorData) const
    {
        double prodValue = 0;
        for (int i = processorData.FirstRowRelativeIndex(); i <= processorData.LastRowRelativeIndex(); ++i)
        {
            int iNetIndex = i - processorData.FirstRowRelativeIndex() + processorData.FirstRowIndex();
            for (int j = 1; j < currentValues.colsCount() - 1; ++j)
            {
                prodValue = prodValue + netModel.xAverageStep(iNetIndex) * netModel.yAverageStep(j) *
                                        currentValues(i, j) * otherValues(i, j);
            }
        }
        return prodValue;
    }


    double MaxNormValue(const DoubleMatrix& currentValues) const
    {
        auto minMax = std::minmax_element(&(currentValues(0,0)),
                        &(currentValues(0,0)) + currentValues.rowsCount() * currentValues.colsCount());
        double min = fabs(*minMax.first);
        double max = fabs(*minMax.second);
        return max > min ? max : min;
    }

};

#endif // DERIVATOR_H
