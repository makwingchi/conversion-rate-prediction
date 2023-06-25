import math

import paddle
import paddle.nn.functional as F


class DNN(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.sparse_feature_dim = self.config["models"]["baseline"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["baseline"]["sparse_inputs_slots"] - 2
        self.layer_sizes = self.config["models"]["baseline"]["fc_sizes"]

        sizes = [self.sparse_feature_dim * self.num_field] + self.layer_sizes + [1]

        self._mlp_layers = []

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

            self._mlp_layers.append(linear)

        self.relu = paddle.nn.ReLU()

    def forward(self, sparse_embs):
        y_dnn = paddle.concat(x=sparse_embs, axis=1)

        for idx, n_layer in enumerate(self._mlp_layers):
            y_dnn = n_layer(y_dnn)

            if idx != len(self._mlp_layers) - 1:
                y_dnn = self.relu(y_dnn)

        return F.sigmoid(y_dnn)
