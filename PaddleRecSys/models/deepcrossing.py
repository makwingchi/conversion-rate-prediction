import math

import paddle
import paddle.nn.functional as F


class DeepCrossingResidualBlock(paddle.nn.Layer):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.linear1 = paddle.nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(input_dim)
                )
            )
        )
        self.linear2 = paddle.nn.Linear(
            in_features=hidden_dim,
            out_features=input_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(hidden_dim)
                )
            )
        )
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.linear1(x))
        out2 = self.linear2(out1) + x

        return self.relu(out2)


class DeepCrossing(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_dim = self.config["models"]["deepcrossing"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["deepcrossing"]["sparse_inputs_slots"] - 2
        self.layer_sizes = self.config["models"]["deepcrossing"]["fc_sizes"]

        self.residual_blocks = []

        for i in range(len(self.layer_sizes)):
            residual_block = DeepCrossingResidualBlock(
                input_dim=self.sparse_feature_dim * self.num_field,
                hidden_dim=self.layer_sizes[i]
            )

            self.add_sublayer('residual_block_%d' % i, residual_block)
            self.residual_blocks.append(residual_block)

        self.final_linear = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(self.sparse_feature_dim * self.num_field)
                )
            )
        )
        self.add_sublayer('final_linear_layer', self.final_linear)

        self.relu = paddle.nn.ReLU()

    def forward(self, sparse_embs):
        x = paddle.concat(sparse_embs, axis=1)
        x = self.relu(x)

        for block in self.residual_blocks:
            x = block(x)

        out = self.final_linear(x)

        return F.sigmoid(out)
