import logging
import sys
import activations as act
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from autoencoder import Autoencoder
from ffnet import FFNet
from layers import FCLayer

plot_figures = False
batch_size = 4
n_layers = 3
eps = 1e-4


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

def create_nn_layers(n_features):
    n_middle_layer_neurons = 32
    shapes = [(n_features, n_middle_layer_neurons),
              (n_middle_layer_neurons,  n_middle_layer_neurons),
              (n_middle_layer_neurons, n_features)]
    activation_functions = [act.LinearActivationFunction(),
                            act.SigmoidActivationFunction(),
                            act.LinearActivationFunction()]
    use_bias_flags = [False, False, False]
    n_layers = len(shapes)
    layers = []
    init_weights = []
    for n_layer in xrange(n_layers):
        layer = FCLayer(shapes[n_layer], activation_functions[n_layer], use_bias_flags[n_layer])
        init_layer_weights = np.random.normal(0, eps, layer.get_params_number())
        init_weights = np.concatenate((init_weights, init_layer_weights))
        layers.append(layer)
    return layers, init_weights

def main():
    try:
        print 'hello'
        batch = create_sample_batch()
        n_features = batch.shape[0]
        layers, init_weights = create_nn_layers(n_features)
        autoEncoder = Autoencoder(layers)
        autoEncoder.net.set_weights(init_weights)
        loss_value, loss_grad, output = autoEncoder.compute_loss(batch)
    except IOError as error:
        print('I/O error({}): {}, {}'.format(error.errno, error.strerror, error.args[0]))
    except AttributeError as error:
        print('Attribute error: {}'.format(error.args[0]))
    except:
        print('Unexpected error: {}'.format(sys.exc_info()[0]))


if __name__ == '__main__':
    main()
