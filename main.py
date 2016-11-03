import logging
import sys
import activations as act
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from autoencoder import Autoencoder
from ffnet import FFNet
from ffnet_creator import FFNetCreator
from layers import FCLayer
from gradient_checker import GradientChecker

plot_figures = False
batch_size = 4
n_layers = 3
eps = 1e-4
gradient_check = True


def create_sample_batch():
    digits = load_digits()
    print(digits.data.shape)
    batch = digits.data[0:batch_size]
    batch = batch.transpose()
    if plot_figures:
        plt.figure(1, figsize=(3, 3))
        plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()
    return batch

def main():
    try:
        batch = create_sample_batch()
        # print batch
        n_features = batch.shape[0]
        net_creator = FFNetCreator()
        if gradient_check:
            gradient_checker = GradientChecker()
            is_correct, left, right = gradient_checker.check_gradient(batch)
            print (is_correct, left, right)
        else:
            layers = net_creator.create_nn_layers_simple(n_features)
            autoencoder = Autoencoder(layers)
            autoencoder.init_weights()
            loss_value, loss_grad, output = autoencoder.compute_loss(batch)
    except IOError as error:
        print('I/O error({}): {}, {}'.format(error.errno, error.strerror, error.args[0]))
    except AttributeError as error:
        print('Attribute error: {}'.format(error.args[0]))
    except:
        print('Unexpected error: {}'.format(sys.exc_info()[0]))


if __name__ == '__main__':
    main()
