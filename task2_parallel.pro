#-------------------------------------------------
#
# Project created by QtCreator 2016-10-12T23:26:10
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = task2_parallel
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += \
    src_parallel/main.cpp \
    src_parallel/ConjugateGradientAlgo.cpp \
    src_parallel/MPIOperations.cpp
core


QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

CONFIG += c++11


INCLUDEPATH += /usr/include/mpich/

LIBS += -lmpich -lopa -lpthread -lrt

QMAKE_CXXFLAGS += -Bsymbolic-functions


HEADERS += \
    src_parallel/NetModel.h \
    src_parallel/ConjugateGradientAlgo.h \
    src_parallel/DifferentialEquationModel.h \
    src_parallel/ApproximateOperations.h \
    src_parallel/Interface.h \
    src_parallel/DoubleMatrix.h \
    src_parallel/ProcessorsData.h \
    src_parallel/MPIOperations.h

