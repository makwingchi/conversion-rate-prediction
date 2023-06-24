import math

import paddle
import paddle.nn.functional as F


class FNN(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_dim = self.config["models"]["fnn"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["fnn"]["sparse_inputs_slots"] - 2
        self.layer_sizes = self.config["models"]["fnn"]["fc_sizes"]

        sizes = [self.sparse_feature_dim * self.num_field] + self.layer_sizes + [1]

        self.nets = []

        for i in range(len(self.layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i])
                    )
                )
            )

            self.nets.append(linear)

        self.relu = paddle.nn.ReLU()

    def forward(self, sparse_embs):
        x = paddle.concat(sparse_embs, axis=1)

        for idx, net in enumerate(self.nets):
            x = net(x)

            if idx != len(self.nets) - 1:
                x = self.relu(x)

        return F.sigmoid(x)
