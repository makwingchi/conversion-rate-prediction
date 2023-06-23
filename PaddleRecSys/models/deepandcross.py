import math

import paddle
import paddle.nn.functional as F


class DeepAndCross(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_dim = self.config["models"]["deepandcross"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["deepandcross"]["sparse_inputs_slots"] - 2
        self.layer_sizes = self.config["models"]["deepandcross"]["fc_sizes"]
        self.num_crosses = self.config["models"]["deepandcross"]["num_crosses"]

        sizes = [self.sparse_feature_dim * self.num_field] + self.layer_sizes

        self.deep_models = []

        for i in range(len(self.layer_sizes)):
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

        self.cross_models = []

        for i in range(self.num_crosses):
            cross_model = paddle.nn.Linear(
                in_features=self.sparse_feature_dim * self.num_field,
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(self.sparse_feature_dim * self.num_field)
                    )
                )
            )

            self.add_sublayer(f"cross_model_{i}", cross_model)
            self.cross_models.append(cross_model)

        self.relu = paddle.nn.ReLU()
        self.final_linear = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field + sizes[-1],
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(self.sparse_feature_dim * self.num_field + sizes[-1])
                )
            )
        )
        self.add_sublayer("final_linear", self.final_linear)

    def forward(self, sparse_embs):
        x = paddle.concat(sparse_embs, axis=1)

        # cross
        x0 = x.unsqueeze(-1)
        xl = x0

        for cross_model in self.cross_models:
            xl_T = paddle.transpose(xl, perm=(0, 2, 1))
            xl = cross_model(paddle.matmul(x0, xl_T)) + xl

        cross_out = xl.squeeze(-1)

        # deep
        deep_out = x
        for deep_model in self.deep_models:
            deep_out = deep_model(deep_out)
            deep_out = self.relu(deep_out)

        concat_out = paddle.concat([cross_out, deep_out], axis=-1)

        return F.sigmoid(self.final_linear(concat_out))
