import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.input_tensor = prediction_tensor
        loss = -np.sum(label_tensor * np.log(prediction_tensor))
        return loss

    def backward(self, label_tensor):
        return -(label_tensor/self.input_tensor)
