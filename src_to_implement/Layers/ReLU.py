from src_to_implement.Layers import Base
import numpy as np


class ReLU(Base.BaseLayer):
    def __init__(self):
        super(ReLU).__init__()
        self.trainable = False
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor
