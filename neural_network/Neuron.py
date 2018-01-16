
import numpy as np


class Neuron:
    def __init__(self, x, value=0):
        self.w = np.full(x, 0.1)
        self.value = value
        self.sum = 0
        self.delta = 0
        #print("\tNew neuron: value=[{}]; w={}".format(value, self.w))

    def activate(self, prev_layers):
        sum = 0
        for i, neuron in enumerate(prev_layers.neurons):
            sum = sum + neuron.value * self.w[i]
        self.sum = sum
        self.value = 1 / (1 + np.exp(-sum))

        return self.value

    def update_delta(self, index, layer):
        combination = 0
        for i, neuron in enumerate(layer.neurons):
            combination += neuron.w[index] * neuron.delta
        self.delta = self.value * (1 - self.value) * combination

    def update_weights(self, prev_layer, learning_rate):
        lr = learning_rate
        for i, neuron in enumerate(prev_layer.neurons):
            self.w[i] += lr * neuron.value * self.delta