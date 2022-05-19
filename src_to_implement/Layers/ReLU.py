from Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        super.__init__()
        self.trainable = True

    def forward(self, input_tensor):
        return np.max(0, input_tensor)

    def backward(self, error_tensor):
        return 1 if error_tensor > 0 else 0
