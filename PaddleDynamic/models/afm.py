import numpy as np

import paddle
import paddle.nn.functional as F


class AFM(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["common"]["num_fields"]
        self.layer_sizes = self.config["models"]["common"]["fc_sizes"]
        self.attention_dim = self.config["models"]["afm"]["attention_dim"]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        triu_indices = np.triu_indices(self.num_field, k=1)
        self.index1 = paddle.to_tensor(triu_indices[0])
        self.index2 = paddle.to_tensor(triu_indices[1])

        self.attention = [
            paddle.nn.Linear(
                in_features=self.sparse_feature_dim,
                out_features=self.attention_dim,
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0))
            ),
            paddle.nn.ReLU(),
            paddle.nn.Linear(
                in_features=self.attention_dim,
                out_features=1,
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=False
            ),
            paddle.nn.Softmax(axis=1)
        ]

        self.attention = paddle.nn.Sequential(*self.attention)

        self.weight_p = paddle.nn.Linear(
            in_features=self.sparse_feature_dim,
            out_features=1,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            bias_attr=False
        )

        self.linear = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field,
            out_features=1,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            bias_attr=False
        )

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature),
                axis=1,
                keepdim=True
            )  # batch x 1 x emb_dim

            feature_ls.append(emb)

        x = paddle.concat(feature_ls, axis=1).astype("float32")  # batch x 26 x emb_dim

        emb1 = paddle.index_select(x, self.index1, axis=1)
        emb2 = paddle.index_select(x, self.index2, axis=1)
        elementwise_product = emb1 * emb2

        attention_weight = self.attention(elementwise_product)
        attention_sum = paddle.sum(attention_weight * elementwise_product, axis=1)  # batch x emb_dim
        attention_out = self.weight_p(attention_sum)  # batch x 1

        flatten_x = paddle.reshape(x, shape=[-1, self.num_field * self.sparse_feature_dim])
        linear_out = self.linear(flatten_x)

        return F.sigmoid(linear_out + attention_out)
