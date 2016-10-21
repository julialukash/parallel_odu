#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of activation functions used within neural networks

from __future__ import print_function
from __future__ import division
import numpy as np


class BaseActivationFunction(object):

    def val(self, inputs):
        """
        Calculates values of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def deriv(self, inputs):
        """
        Calculates first derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def second_deriv(self, inputs):
        """
        Calculates second derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')


class LinearActivationFunction(BaseActivationFunction):
    pass


class SigmoidActivationFunction(BaseActivationFunction):
    pass


class ReluActivationFunction(BaseActivationFunction):
    pass

