import logging
import sys
import activations as act
import layers
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from autoencoder import Autoencoder
from ffnet import FFNet

plot_figures = False
batch_size = 4

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
        print 'hello'
        batch = create_sample_batch()
        print batch

        activation_function = act.SigmoidActivationFunction()
        print activation_function.val(batch)
        # layers = []
        # autoEncoder = Autoencoder(layers)
    except IOError as error:
        print('I/O error({}): {}, {}'.format(error.errno, error.strerror, error.args[0]))
    except AttributeError as error:
        print('Attribute error: {}'.format(error.args[0]))
    except:
        print('Unexpected error: {}'.format(sys.exc_info()[0]))


if __name__ == '__main__':
    main()
