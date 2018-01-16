from neural_network.Neuron import Neuron
import numpy as np


class Layer:
    def __init__(self, weights_len, num_neurons, values):
        self.num_neurons = num_neurons
        self.neurons = []
        self.values = values
        self.weights_len = weights_len
        self.set_neurons(num_neurons, values)

    def add_neuron(self, weights_len, value):
        self.neurons.append(Neuron(weights_len, value=value))

    def set_neurons(self, num_neurons, values):
        for i in range(num_neurons):
            self.add_neuron(self.weights_len, values[i])

    def update_neuron_value(self, prev_layer):
        for i, neuron in enumerate(self.neurons):
            neuron.activate(prev_layer)
            # print("\t\tNeuron: ", i)
            # print("\t\t\t Sum: ", neuron.sum)
            # print("\t\t\t value: ", neuron.value)

    def update_neurons_delta(self, next_layer):
        for i, neuron in enumerate(self.neurons):
            neuron.update_delta(i, next_layer)
            #print("\t\tNeuron[{}]: D = {}".format(i, neuron.delta))

    def update_neuron_delta_with_neuron(self, neuron):
        for i, layer_neuron in enumerate(self.neurons):
            layer_neuron.delta = layer_neuron.value * (1 - layer_neuron.value) * neuron.w[i] * neuron.delta

    def update_neuron_weights(self, prev_layer, learning_rate):
        for i, neuron in enumerate(self.neurons):
            neuron.update_weights(prev_layer, learning_rate)
            #print("\t\tNeuron[{}]: W = {}".format(i, neuron.w))
