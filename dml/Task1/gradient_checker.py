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

    def check_gradient_img(self, layers, inputs):
        n_objects = inputs.shape[1]
        autoencoder = Autoencoder(layers)
        p_vector = self.net_creator.create_p_vector(autoencoder.net.params_number)

        autoencoder.init_weights()
        loss_value, loss_grad, output = autoencoder.compute_loss(inputs)
        left = loss_grad.transpose().dot(p_vector) / n_objects

        biased_autoencoder = Autoencoder(layers)
        weights = autoencoder.net.get_weights()
        biased_weights = weights + 1j * self.eps * p_vector
        biased_autoencoder.net.set_weights(biased_weights)

        biased_loss_value, biased_loss_grad, biased_output = biased_autoencoder.compute_loss(inputs)
        right = biased_loss_value.imag / self.eps
        diff = abs(left - right)
        return diff < self.eps, left, right

    def run_tests(self, inputs):
        n_features = inputs.shape[0]
        layers = self.net_creator.create_nn_three_layers_simple(n_features)
        is_correct, left, right = self.check_gradient_img(layers, inputs)
        print(is_correct, left, right)
        layers = self.net_creator.create_nn_five_layers_simple(n_features)
        is_correct, left, right = self.check_gradient_img(layers, inputs)
        print(is_correct, left, right)
        return


    def check_rp_img(self, layers, inputs):
        n_objects = inputs.shape[1]
        autoencoder = Autoencoder(layers)
        p_vector = self.net_creator.create_p_vector(autoencoder.net.params_number)

        autoencoder.init_weights()
        autoencoder.compute_loss(inputs)
        loss_Rp_grad  = autoencoder.compute_hessvec(p_vector)
        left = loss_Rp_grad

        biased_autoencoder = Autoencoder(layers)
        weights = autoencoder.net.get_weights()
        biased_weights = weights + 1j * self.eps * p_vector
        biased_autoencoder.net.set_weights(biased_weights)

        biased_loss_value, biased_loss_grad, biased_output = biased_autoencoder.compute_loss(inputs)
        right = np.imag(biased_loss_grad)  / self.eps
        is_equal = np.allclose(left, right, self.eps)
        print(is_equal, np.sum(left), np.sum(right))
        return is_equal

    def run_rp_tests(self, inputs):
        n_features = inputs.shape[0]
        layers = self.net_creator.create_nn_three_layers_simple(n_features)
        is_correct = self.check_rp_img(layers, inputs)
        layers = self.net_creator.create_nn_five_layers_simple(n_features)
        is_correct = self.check_rp_img(layers, inputs)
        layers = self.net_creator.create_nn_three_layers_linear(n_features)
        is_correct = self.check_rp_img(layers, inputs)
        return


    def check_gaussnewtonvec(self, layers, inputs):
        n_objects = inputs.shape[1]
        autoencoder = Autoencoder(layers)
        p_vector = self.net_creator.create_p_vector(autoencoder.net.params_number)

        autoencoder.init_weights()
        autoencoder.compute_loss(inputs)
        loss_Rp_grad  = autoencoder.compute_hessvec(p_vector)
        left = loss_Rp_grad

        Gp = autoencoder.compute_gaussnewtonvec(p_vector)
        right = Gp
        is_equal = np.allclose(left, right, self.eps)
        print(is_equal, np.sum(left), np.sum(right))
        return is_equal

    def run_gaussnewtonvec_tests(self, inputs):
        n_features = inputs.shape[0]
        layers = self.net_creator.create_nn_three_layers_linear(n_features)
        is_correct = self.check_gaussnewtonvec(layers, inputs)
        return