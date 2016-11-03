
import numpy as np
import activations as act
from layers import FCLayer

class FFNetCreator:
     def create_nn_layers_simple(self, n_features):
        np.random.seed(1984)
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
        for n_layer in xrange(n_layers):
            layer = FCLayer(shapes[n_layer], activation_functions[n_layer], use_bias_flags[n_layer])
            layers.append(layer)
        return layers

     def create_p_vector(self, n_values):
         # p_vector = np.ones((1, len(weights))).flatten()
         return np.random.rand(1, n_values).flatten()