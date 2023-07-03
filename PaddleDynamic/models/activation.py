import numpy as np

import paddle
import paddle.nn.functional as F


class Dice(paddle.nn.Layer):
    def __init__(self, input_dim, eps=1e-9):
        super(Dice, self).__init__()
        self.bn = paddle.nn.BatchNorm1D(input_dim, epsilon=eps, momentum=0.01)

        indices = np.zeros(input_dim)
        w_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(value=indices))
        self.alpha = paddle.create_parameter(shape=indices.shape, dtype="float32", attr=w_attr)

    def forward(self, X):
        p = F.sigmoid(self.bn(X))
        output = p * X + self.alpha * (1 - p) * X
        return output
