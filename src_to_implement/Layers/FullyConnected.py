from src_to_implement.Layers import Base
import numpy as np
import random


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super.__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(input_size, output_size)
        self._optimizer = None
        self.gradient_weights = None

    def forward(self, input_tensor):
        output = np.dot(input_tensor, self.weights)
        return output

    def backward(self, error_tensor):
        pass

    # getter method
    @property
    def optimizer(self):
        return self._optimizer

    # setter method
    @optimizer.setter
    def optimizer(self, x):
        self._optimizer = x

    # getter method
    @property
    def gradient_weights(self):
        return self.gradient_weights

    # setter method
    @gradient_weights.setter
    def gradient_weights(self, x):
        self.gradient_weights = x
