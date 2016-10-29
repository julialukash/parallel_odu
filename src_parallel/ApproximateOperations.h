#ifndef DERIVATOR_H
#define DERIVATOR_H

#include "Interface.h"
#include "ProcessorsData.h"
#include <algorithm>

//#define DEBUG_MODE = 1

class ApproximateOperations
{
private:
    std::shared_ptr<NetModel> netModel;
public:
    ApproximateOperations(std::shared_ptr<NetModel> model)
    {
        netModel = model;
    }

    // note: calculates -laplass(currentValues)
    DoubleMatrix CalculateLaplass(const DoubleMatrix& currentValues, std::shared_ptr<ProcessorsData> processorDataPtr)
    {
        auto startIndex = processorDataPtr->IsFirstProcessor() ? 2 : 1;
        auto endIndex = processorDataPtr->IsLastProcessor() ? processorDataPtr->RowsCountWithBorders() - 3 : processorDataPtr->RowsCountWithBorders() - 2;
#ifdef DEBUG_MODE
        std::cout <<"startIndex = " << startIndex << ", endIndex = " << endIndex << ", currentValues = \n" << currentValues << std::endl;
#endif
        auto laplassValues = DoubleMatrix(processorDataPtr->RowsCountWithBorders(), netModel->xPointsCount);
        for (auto i = startIndex; i <= endIndex; ++i)
        {
            auto iNetIndex = i - startIndex + processorDataPtr->FirstRowIndex();
#ifdef DEBUG_MODE
        std::cout <<"i = " << i << ", iNetIndex = " << iNetIndex << std::endl;
#endif
            for (auto j = 1; j < laplassValues.colsCount() - 1; ++j)
            {
                auto xPart = (currentValues(i, j) - currentValues(i - 1, j)) / netModel->xStep(iNetIndex - 1) -
                             (currentValues(i + 1, j) - currentValues(i, j)) / netModel->xStep(iNetIndex);
                auto yPart = (currentValues(i, j) - currentValues(i, j - 1)) / netModel->yStep(j - 1) -
                             (currentValues(i, j + 1) - currentValues(i, j)) / netModel->yStep(j);
                laplassValues(i, j) = xPart / netModel->xAverageStep(iNetIndex) + yPart / netModel->yAverageStep(j);
            }
        }
#ifdef DEBUG_MODE
        std::cout <<"laplassValues = \n" << laplassValues << std::endl;
#endif
        return laplassValues;
    }

    double ScalarProduct(const DoubleMatrix& currentValues, const DoubleMatrix& otherValues,  std::shared_ptr<ProcessorsData> processorDataPtr)
    {
        auto startIndex = processorDataPtr->IsFirstProcessor() ? 2 : 1;
        auto endIndex = processorDataPtr->IsLastProcessor() ? processorDataPtr->RowsCountWithBorders() - 3 : processorDataPtr->RowsCountWithBorders() - 2;

        double prodValue = 0;
        for (auto i = startIndex; i <= endIndex; ++i)
        {
            auto iNetIndex = i - startIndex + processorDataPtr->FirstRowIndex();
            for (auto j = 1; j < currentValues.colsCount() - 1; ++j)
            {
                prodValue = prodValue + netModel->xAverageStep(iNetIndex) * netModel->yAverageStep(j) *
                                        currentValues(i, j) * otherValues(i, j);
            }
        }
        return prodValue;
    }

//    double NormValueEq(const DoubleMatrix& currentValues)
//    {
//        return sqrt(ScalarProduct(currentValues, currentValues));
//    }


    double MaxNormValue(const DoubleMatrix& currentValues)
    {
        return fabs(*std::max_element(&(currentValues.matrix[0][0]),
                &(currentValues.matrix[0][0]) + currentValues.rowsCount() * currentValues.colsCount()));
    }

};

#endif // DERIVATOR_H
