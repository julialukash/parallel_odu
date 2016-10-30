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
    std::shared_ptr<DoubleMatrix> CalculateLaplass(const DoubleMatrix& currentValues, const ProcessorsData& processorDataPtr)
    {
        int startIndex = processorDataPtr.IsFirstProcessor() ? 2 : 1;
        int endIndex = processorDataPtr.IsLastProcessor() ? processorDataPtr.RowsCountWithBorders() - 3 : processorDataPtr.RowsCountWithBorders() - 2;
#ifdef DEBUG_MODE
        std::cout <<"ApproximateOperations.CalculateLaplass currentValues = \n" << currentValues << std::endl;
        std::cout <<"ApproximateOperations.CalculateLaplass startIndex = " << startIndex << ", endIndex = " << endIndex << std::endl;
#endif
        auto laplassValues = std::make_shared<DoubleMatrix>(processorDataPtr.RowsCountWithBorders(), netModel->xPointsCount);
#ifdef DEBUG_MODE
        std::cout <<"ApproximateOperations.CalculateLaplass laplassValues = \n" << laplassValues << std::endl;
#endif
        for (int i = startIndex; i <= endIndex; ++i)
        {
            int iNetIndex = i - startIndex + processorDataPtr.FirstRowIndex();

            for (int j = 1; j < laplassValues->colsCount() - 1; ++j)
            {
                double xPart = (currentValues(i, j) - currentValues(i - 1, j)) / netModel->xStep(iNetIndex - 1) -
                             (currentValues(i + 1, j) - currentValues(i, j)) / netModel->xStep(iNetIndex);
                double yPart = (currentValues(i, j) - currentValues(i, j - 1)) / netModel->yStep(j - 1) -
                             (currentValues(i, j + 1) - currentValues(i, j)) / netModel->yStep(j);
                (*laplassValues)(i, j) = xPart / netModel->xAverageStep(iNetIndex) + yPart / netModel->yAverageStep(j);
#ifdef DEBUG_MODE
                std::cout <<"ApproximateOperations.CalculateLaplass i = " << i << ", iNetIndex = " << iNetIndex << ", j = " << j << ", value = " << laplassValues(i, j) << std::endl;
#endif
            }
        }
#ifdef DEBUG_MODE
        std::cout <<"ApproximateOperations.CalculateLaplass laplassValues = \n" << laplassValues << std::endl;
#endif
        return laplassValues;
    }

    double ScalarProduct(const DoubleMatrix& currentValues, const DoubleMatrix& otherValues,  std::shared_ptr<ProcessorsData> processorDataPtr)
    {
        int startIndex = processorDataPtr->IsFirstProcessor() ? 2 : 1;
        int endIndex = processorDataPtr->IsLastProcessor() ? processorDataPtr->RowsCountWithBorders() - 3 : processorDataPtr->RowsCountWithBorders() - 2;

        double prodValue = 0;
        for (int i = startIndex; i <= endIndex; ++i)
        {
            int iNetIndex = i - startIndex + processorDataPtr->FirstRowIndex();
            for (int j = 1; j < currentValues.colsCount() - 1; ++j)
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
        auto minMax = std::minmax_element(&(currentValues(0,0)),
                        &(currentValues(0,0)) + currentValues.rowsCount() * currentValues.colsCount());
        double min = fabs(*minMax.first);
        double max = fabs(*minMax.second);
#ifdef DEBUG_MODE
        std::cout << "MaxNormValue min = " << min << ", max = " << max << std::endl;
#endif
        return max > min ? max : min;
    }

};

#endif // DERIVATOR_H
