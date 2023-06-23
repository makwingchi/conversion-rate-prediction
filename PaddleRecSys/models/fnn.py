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
        acts = ["tanh" for _ in range(len(self.layer_sizes))] + [None]

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

            self.add_sublayer(f"linear_layer_{i}", linear)
            self.nets.append(linear)

            if acts[i] == "tanh":
                act = paddle.nn.Tanh()
                self.add_sublayer(f"tanh_{i}", act)
                self.nets.append(act)

    def forward(self, sparse_embs):
        x = paddle.concat(sparse_embs, axis=1)

        for net in self.nets:
            x = net(x)

        return F.sigmoid(x)
