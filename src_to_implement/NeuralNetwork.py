import numpy as np
from src_to_implement.Optimization import Optimizers
import copy


class NeuralNetwork:
    def __init__(self, optimizers: Optimizers):
        self.optimizers = optimizers
        self.loss = []
        self.layers = []
        self.label_tensor = None
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        output_layer = np.copy(input_tensor)
        for layer in self.layers:
            output_layer = layer.forward(output_layer)
        return self.loss_layer.forward(output_layer, self.label_tensor)

    def backward(self):
        output_back = self.loss_layer.backward(self.label_tensor)
        for back_layer in self.layers[::-1]:
            output_back = back_layer.backward(output_back)

    def append_layer(self, layer):
        if layer.trainable:
            optimizer = copy.deepcopy(self.optimizers)
            layer.optimizer = optimizer
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            running_loss = self.forward()
            self.backward()
            self.loss.append(running_loss)

    def test(self, input_tensor):
        output_layer = input_tensor
        for layer in self.layers:
            output_layer = layer.forward(output_layer)
        return output_layer
