#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of layers used within neural networks

from __future__ import print_function
from __future__ import division
import numpy as np


class BaseLayer(object):

    def get_params_number(self):
        """
        :return num_params: number of parameters used in layer
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_weights(self):
        """
        :return w: current layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_weights(self, w):
        """
        Takes weights as a one-dimensional numpy vector and assign them to layer parameters in convenient shape,
        e.g. matrix shape for fully-connected layer
        :param w: layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_direction(self, p):
        """
        Takes direction vector as a one-dimensional numpy vector and assign it to layer parameters direction vector
        in convenient shape, e.g. matrix shape for fully-connected layer
        :param p: layer parameters direction vector, numpy vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def forward(self, inputs):
        """
        Forward propagation for layer. Intermediate results are saved within layer parameters.
        :param inputs: input batch, numpy matrix of size num_inputs x num_objects
        :return outputs: layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def backward(self, derivs):
        """
        Backward propagation for layer. Intermediate results are saved within layer parameters.
        :param derivs: loss derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_derivs: loss derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_derivs: loss derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_forward(self, Rp_inputs):
        """
        Rp forward propagation for layer. Intermediate results are saved within layer parameters.
        :param Rp_inputs: Rp input batch, numpy matrix of size num_inputs x num_objects
        :return Rp_outputs: Rp layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_backward(self, Rp_derivs):
        """
        Rp backward propagation for layer.
        :param Rp_derivs: loss Rp derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_Rp_derivs: loss Rp derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_Rp_derivs: loss Rp derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_activations(self):
        """
        :return outputs: activations computed in forward pass, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')


class FCLayer(BaseLayer):

    def __init__(self, shape, afun, use_bias=False):
        """
        :param shape: layer shape, a tuple (num_inputs, num_outputs)
        :param afun: layer activation function, instance of BaseActivationFunction
        :param use_bias: flag for using bias parameters
        """
        self.shape = shape
        self.afun = afun
        self.use_bias = use_bias
        self.shape_bias = (shape[0] + 1, shape[1]) if self.use_bias else shape


    def get_params_number(self):
        """
        :return num_params: number of parameters used in layer
        """
        return self.shape_bias[0] * self.shape_bias[1]

    def get_weights(self):
        """
        :return w: current layer weights as a numpy one-dimensional vector
        """
        return self.weights.flatten()

    def set_weights(self, w):
        """
        Takes weights as a one-dimensional numpy vector and assign them to layer parameters in convenient shape,
        e.g. matrix shape for fully-connected layer
        :param w: layer weights as a numpy one-dimensional vector
        """
        if w.shape[0] != self.get_params_number():
            raise AttributeError('Incorrect weight dimension, layer shape {}, given number of params = {}'.format(self.shape_bias, w.shape[0]))
        self.weights = np.reshape(w, (self.shape_bias[1], self.shape_bias[0]))

    def set_direction(self, p):
        """
        Takes direction vector as a one-dimensional numpy vector and assign it to layer parameters direction vector
        in convenient shape, e.g. matrix shape for fully-connected layer
        :param p: layer parameters direction vector, numpy vector
        """
        if p.shape[0] != self.get_params_number():
            raise AttributeError('Incorrect weight dimension, layer shape {}, given number of params = {}'.format(self.shape_bias, p.shape[0]))
        self.p = np.reshape(p, (self.shape_bias[1], self.shape_bias[0]))

    def forward(self, inputs):
        """
        Forward propagation for layer. Intermediate results are saved within layer parameters.
        :param inputs: input batch, numpy matrix of size num_inputs x num_objects
        :return outputs: layer activations, numpy matrix of size num_outputs x num_objects
        """
        n_objects = inputs.shape[1]
        if self.use_bias:
            inputs = np.vstack((inputs, np.ones((1, n_objects))))
        self.inputs = inputs
        self.u = self.weights.dot(inputs)
        self.z =  self.afun.val(self.u)
        return self.z

    def backward(self, derivs):
        """
        Backward propagation for layer. Intermediate results are saved within layer parameters.
        :param derivs: loss derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_derivs: loss derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_derivs: loss derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        self.derivs = derivs
        self.input_derivs = derivs * self.afun.deriv(self.u)
        self.w_derivs = self.input_derivs.dot(self.inputs.transpose())
        self.output_derivs = self.weights.transpose().dot(self.input_derivs)
        if self.use_bias:
            self.output_derivs = np.delete(self.output_derivs, (self.output_derivs.shape[0] - 1), axis=0)
        return self.output_derivs, self.w_derivs.flatten()

    def Rp_forward(self, Rp_inputs):
        """
        Rp forward propagation for layer. Intermediate results are saved within layer parameters.
        :param Rp_inputs: Rp input batch, numpy matrix of size num_inputs x num_objects
        :return Rp_outputs: Rp layer activations, numpy matrix of size num_outputs x num_objects
        """
        n_objects = Rp_inputs.shape[1]
        if self.use_bias:
            Rp_inputs = np.vstack((Rp_inputs, np.zeros(1, n_objects)))
        self.rp_inputs = Rp_inputs
        self.rp_u = self.weights.dot(self.rp_inputs) + self.p.dot(self.inputs)
        self.rp_z = self.afun.deriv(self.u) * self.rp_u
        return self.rp_z

    def Rp_backward(self, Rp_derivs):
        """
        Rp backward propagation for layer.
        :param Rp_derivs: loss Rp derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_Rp_derivs: loss Rp derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_Rp_derivs: loss Rp derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        self.rp_input_derivs = Rp_derivs * self.afun.deriv(self.u) + self.derivs * self.afun.second_deriv(self.u) * self.rp_u
        self.rp_w_derivs = self.rp_input_derivs.dot(self.inputs.transpose()) + self.input_derivs.dot(self.rp_inputs.transpose())
        self.rp_output_derivs = self.p.transpose().dot(self.input_derivs) + self.weights.transpose().dot(self.rp_input_derivs)
        if self.use_bias:
            self.rp_output_derivs = np.delete(self.rp_output_derivs, (self.rp_output_derivs.shape[0] - 1), axis=0)
        return self.rp_output_derivs, self.rp_w_derivs.flatten()

    def get_activations(self):
        """
        :return outputs: activations computed in forward pass, numpy matrix of size num_outputs x num_objects
        """
        return self.u

