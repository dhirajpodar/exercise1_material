import Base
import numpy as np

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super.__init__()
        self.trainable = True

    def forward(self, input_size):
        output = np.exp(input_size)
        y_pred = output / np.sum(output, axis=1, keepdims=True)

        return y_pred

    def backward(self):
        pass
