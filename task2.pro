#-------------------------------------------------
#
# Project created by QtCreator 2016-10-12T23:26:10
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = task2
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += \
    src/main.cpp \
    src/ConjugateGradientAlgo.cpp
core


QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

CONFIG += c++11

HEADERS += \
    src/NetModel.h \
    src/ConjugateGradientAlgo.h \
    src/DifferentialEquationModel.h \
    src/ApproximateOperations.h \
    src/Interface.h

