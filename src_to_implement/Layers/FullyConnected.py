from src_to_implement.Layers import Base
import numpy as np


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super(FullyConnected).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.rand(input_size, output_size)
        self._optimizer = None
        self.gradient = None
        self.input_tensor = None
        self.error_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output = np.dot(input_tensor, self.weights)
        return output

    def backward(self, error_tensor):
        self.error_tensor = np.dot(error_tensor, self.weights.T)
        self.gradient = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient)
        return self.error_tensor

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
        return self.gradient

    # setter method
    @gradient_weights.setter
    def gradient_weights(self, x):
        self.gradient = x
