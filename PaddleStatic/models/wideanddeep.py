import math

import paddle
import paddle.nn.functional as F


class WideAndDeep(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_dim = self.config["models"]["wideanddeep"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["wideanddeep"]["sparse_inputs_slots"] - 2
        self.layer_sizes = self.config["models"]["wideanddeep"]["fc_sizes"]

        sizes = [self.sparse_feature_dim * self.num_field] + self.layer_sizes + [1]

        self.wide_model = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(self.sparse_feature_dim * self.num_field)
                )
            )
        )

        self.deep_models = []
        for i in range(len(self.layer_sizes) + 1):
            deep_model = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i])
                    )
                )
            )

            self.deep_models.append(deep_model)

        self.relu = paddle.nn.ReLU()

    def forward(self, sparse_embs):
        x = paddle.concat(sparse_embs, axis=1)

        wide_out = self.wide_model(x)
        deep_out = x

        for idx, layer in enumerate(self.deep_models):
            deep_out = layer(deep_out)

            if idx != len(self.deep_models) - 1:
                deep_out = self.relu(deep_out)

        return F.sigmoid(wide_out + deep_out)
