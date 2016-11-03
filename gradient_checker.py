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

    def check_gradient(self, inputs):
        n_features = inputs.shape[0]
        n_objects = inputs.shape[1]
        layers = self.net_creator.create_nn_layers_simple(n_features)

        autoencoder = Autoencoder(layers)
        autoencoder.init_weights()
        weights = autoencoder.net.get_weights()
        loss_value, loss_grad, output = autoencoder.compute_loss(inputs)

        # p_vector = np.ones((1, len(weights))).flatten()
        p_vector = np.random.rand(1, len(weights)).flatten()

        right = loss_grad.transpose().dot(p_vector) / n_objects

        eps = 1e-10
        biased_weights = weights + 1j * eps * p_vector
        # biased_weights = biased_weights.flatten()
        biased_autoencoder = Autoencoder(layers)
        biased_autoencoder.net.set_weights(biased_weights)
        biased_loss_value, biased_loss_grad, biased_output = biased_autoencoder.compute_loss(inputs)
        left = biased_loss_value.imag / eps
        diff = abs(left - right)
        return diff < eps, left, right
