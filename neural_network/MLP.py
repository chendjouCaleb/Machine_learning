from neural_network.Layer import Layer
from neural_network.Neuron import Neuron
import numpy as np


class MLP:
    def __init__(self, num_hidden_layers, learning_rate=0.001, epoch=10000):
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.layers = []
        self.first_layer = 0
        self.output = Neuron(num_hidden_layers[-1])
        self.features = np.array([])
        self.targets = np.array([])

    def fit(self, features, targets):
        self.features = features
        self.targets = targets
        self.first_layer = Layer(1, len(features[0]), features[0])
        self.layers.append(self.first_layer)
        self.set_layers()
        last_layer = Layer(self.layers[-1].num_neurons, 1, [0])
        self.layers.append(last_layer)
        print("Network has: {} layers".format(len(self.layers)))

    def train(self):
        for i in range(self.epoch):
            for feature, target in zip(self.features, self.targets):
                print("X = {} Y = {}".format(feature, target))
                self.propagate(feature)
                self.back_propagation(target)
                self.update_weights()
                print("Output: [{}]\n".format(self.layers[-1].neurons[0].value))

    def propagate(self, feature):
        self.layers[0].values = feature
        for i, layer in enumerate(self.layers[1:]):
            # print("\t Layer: ", i + 1)
            layer.update_neuron_value(self.layers[i])


    def update_weights(self):
        # print("Update weights")
        for i, layer in enumerate(self.layers[1:]):
            # print("\t Layer: ", i + 1)
            layer.update_neuron_weights(self.layers[i], self.learning_rate)

    def back_propagation(self, target):
        # print("Back propagation")

        self.update_last_layer_delta([target])
        hidden_layers = self.layers[1:len(self.layers)-1]
        # print("Hidden Layer: {}".format(len(hidden_layers)))
        j = len(self.layers)
        for i, layer in enumerate(reversed(hidden_layers)):
            # print("\tLayer[{}] with {} Neurons".format((j - i - 2), len(layer.neurons)))
            # print("\tLayer[{}] with {} Neurons".format((j - i - 1), len(self.layers[j-i-1].neurons)))
            layer.update_neurons_delta(self.layers[j-i-1])

    def update_last_layer_delta(self, targets):
        for i, (neuron, target) in enumerate(zip(self.layers[-1].neurons, targets)):
            delta = target - neuron.value
            neuron.delta = delta
            # print("\t\tNeuron[{}] : Delta=[{}]".format(i, delta))

    def add_layer(self, weights_len, num_neuron):
        # print("New Layers: ", num_neuron, " neurons")
        self.layers.append(Layer(weights_len, num_neuron, np.full(num_neuron, 0.1)))

    def set_layers(self):
        self.add_layer(len(self.features[0]), self.num_hidden_layers[0])
        for i, hidden_layer in enumerate(self.num_hidden_layers[1:]):
            self.add_layer(self.num_hidden_layers[i], hidden_layer)

    @staticmethod
    def predict(prob):
        if prob >= 0.5:
            return 1
        return 0
