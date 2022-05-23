from src_to_implement.Layers import Base
import numpy as np


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super(SoftMax).__init__()
        self.trainable = False
        self.y_prediction = None
        self.error_tensor = None

    def forward(self, input_size):
        exp_output = np.exp(input_size - np.max(input_size, axis=1, keepdims=True))
        self.y_prediction = exp_output / np.sum(exp_output, axis=1, keepdims=True)

        return self.y_prediction

    def backward(self, error_tensor):
        softmax_back = self.y_prediction-(np.sum(error_tensor*self.y_prediction, axis=1, keepdims=True))
        return softmax_back
