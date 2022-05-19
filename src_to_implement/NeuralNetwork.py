import numpy as np
from src_to_implement.Optimization import Optimizers


class NeuralNetwork:
    def __init__(self, optimizers: Optimizers):
        self.optimizers = optimizers
        self.loss = None
        self.layers = None
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        pass

    def backward(self):
        pass

    def append_layer(self):
        pass

    def train(self, iterations):
        pass

    def test(self, input_tensor):
        pass
