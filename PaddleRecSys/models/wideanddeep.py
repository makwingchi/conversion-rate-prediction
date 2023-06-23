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
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]

        self.wide_model = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(self.sparse_feature_dim * self.num_field)
                )
            )
        )
        self.add_sublayer("wide_model", self.wide_model)

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

            self.add_sublayer(f"deep_model_{i}", deep_model)
            self.deep_models.append(deep_model)

            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('relu_%d' % i, act)
                self.deep_models.append(act)

    def forward(self, sparse_embs):
        x = paddle.concat(sparse_embs, axis=1)

        wide_out = self.wide_model(x)
        deep_out = x

        for layer in self.deep_models:
            deep_out = layer(deep_out)

        return F.sigmoid(wide_out + deep_out)
