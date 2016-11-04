from __future__ import print_function
from __future__ import division

from autoencoder import Autoencoder
from ffnet_creator import FFNetCreator
from layers import FCLayer
import activations as act
import ffnet
import numpy as np


class GradientChecker:
    def __init__(self):
        self.net_creator = FFNetCreator()
        self.eps = 1e-10

    def check_gradient(self, inputs):
        n_features = inputs.shape[0]
        n_objects = inputs.shape[1]
        layers = self.net_creator.create_nn_three_layers_simple(n_features)

        autoencoder = Autoencoder(layers)
        autoencoder.init_weights()
        weights = autoencoder.net.get_weights()
        loss_value, loss_grad, output = autoencoder.compute_loss(inputs)

        p_vector = self.net_creator.create_p_vector(autoencoder.net.params_number)

        left = loss_grad.transpose().dot(p_vector) / n_objects

        biased_weights = weights + 1j * self.eps * p_vector
        biased_autoencoder = Autoencoder(layers)
        biased_autoencoder.net.set_weights(biased_weights)
        biased_loss_value, biased_loss_grad, biased_output = biased_autoencoder.compute_loss(inputs)
        right = biased_loss_value.imag / self.eps
        diff = abs(left - right)
        return diff < self.eps, left, right


    def check_hessvec(self, inputs):
        pass
        # n_features = inputs.shape[0]
        # n_objects = inputs.shape[1]
        # layers = self.net_creator.create_nn_three_layers_simple(n_features)
        #
        # autoencoder = Autoencoder(layers)
        # autoencoder.init_weights()
        # weights = autoencoder.net.get_weights()
        # loss_value, loss_grad, output = autoencoder.compute_loss(inputs)
        #
        # p_vector = self.net_creator.create_p_vector(autoencoder.net.params_number)
        #
        # loss_Rp_grad, Rp_L, Rp_outputs = autoencoder.compute_hessvec(p_vector)
        #
        # left = loss_Rp_grad
        #
        # biased_weights = weights + 1j * self.eps * p_vector
        # biased_autoencoder = Autoencoder(layers)
        # biased_autoencoder.net.set_weights(biased_weights)
        # biased_loss_value, biased_loss_grad, biased_output = biased_autoencoder.compute_loss(inputs)
        # right = biased_loss_grad.imag / self.eps
        # diff = abs(left - right)
        # print (left, right, diff, diff.sum())
        # return diff < self.eps, left, right
