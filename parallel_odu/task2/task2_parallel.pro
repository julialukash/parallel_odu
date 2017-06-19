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
    src_parallel/conjugate_gradient_algo.cpp \
    src_parallel/mpi_operations.cpp
core


QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp


CONFIG += c++11

INCLUDEPATH += /usr/include/mpich/

LIBS += -lmpich -lopa -lpthread -lrt

QMAKE_CXXFLAGS += -Bsymbolic-functions


HEADERS += \
    src_parallel/approximate_operations.h \
    src_parallel/conjugate_gradient_algo.h \
    src_parallel/differential_equation_model.h \
    src_parallel/double_matrix.h \
    src_parallel/interface.h \
    src_parallel/mpi_operations.h \
    src_parallel/net_model.h \
    src_parallel/processors_data.h
