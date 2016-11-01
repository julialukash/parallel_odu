#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of autoencoder using general feed-forward neural network

from __future__ import print_function
from __future__ import division
import ffnet
import numpy as np


class Autoencoder:

    def __init__(self, layers):
        """
        :param layers: a list of fully-connected layers
        """
        self.net = ffnet.FFNet(layers)

        if self.net.layers[0].shape[0] != self.net.layers[-1].shape[1]:
            raise ValueError('In the given autoencoder number of inputs and outputs is different!')

    def compute_loss_function(self, inputs, outputs):
        n_objects = inputs.shape[1]
        sum = np.sum(inputs - outputs)
        return sum / n_objects

    def compute_loss(self, inputs):
        """
        Computes autoencoder loss value and loss gradient using given batch of data
        :param inputs: numpy matrix of size num_features x num_objects
        :return loss: loss value, a number
        :return loss_grad: loss gradient, numpy vector of length num_params
        """
        outputs = self.net.compute_outputs(inputs)
        # compute loss deriv on outputs
        last_layer_output_derivs = outputs - inputs
        loss_deriv = last_layer_output_derivs
        loss_value = self.compute_loss_function(inputs, outputs)
        loss_grad = self.net.compute_loss_grad(loss_deriv)
        return loss_value, loss_grad, outputs

    def compute_hessvec(self, p):
        """
        Computes a product of Hessian and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Hp: a numpy vector of length num_params
        """
        pass

    def compute_gaussnewtonvec(self, p):
        """
        Computes a product of Gauss-Newton Hessian approximation and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Gp: a numpy vector of length num_params
        """
        pass

    def run_adam(self, inputs, step_size=0.1, max_epoch=300, minibatch_size=20, l2_coef=1e-5, test_inputs=None):
        """
        ADAM stochastic optimization method with fixed stepsizes
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param max_epoch: maximal number of epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numby matrix of size num_features x num_test_objects
        """
        raise NotImplementedError('Implementation will be provided')

    def run_hfn(self, inputs):
        """
        Hessian-free Newton optimization method
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        """
        raise NotImplementedError('Implementation will be provided')
