import numpy as np

import paddle
import paddle.nn.functional as F


MIN_FLOAT = np.finfo(np.float32).min / 100.0


class DLRM(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["common"]["num_fields"]
        self.layer_sizes = self.config["models"]["common"]["fc_sizes"]
        self.self_interaction = self.config["models"]["dlrm"]["self_interaction"]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        if self.self_interaction:
            concat_size = int((self.num_field + 1) * self.num_field / 2)
        else:
            concat_size = int((self.num_field - 1) * self.num_field / 2)

        sizes = [concat_size] + self.layer_sizes + [1]
        self.layers = []

        for i in range(len(sizes) - 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i+1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0))
            )

            self.layers.append(linear)

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature),
                axis=1
            )

            feature_ls.append(emb)

        x = paddle.concat(feature_ls, axis=1).astype("float32")

        T = paddle.reshape(x, shape=[-1, self.num_field, self.sparse_feature_dim])
        Z = paddle.bmm(T, paddle.transpose(T, perm=[0, 2, 1]))

        if self.self_interaction:
            Zflat = paddle.triu(Z, 0) + paddle.tril(paddle.ones_like(Z) * MIN_FLOAT, -1)
        else:
            Zflat = paddle.triu(Z, 1) + paddle.tril(paddle.ones_like(Z) * MIN_FLOAT, 0)

        Zflat = paddle.masked_select(Zflat, paddle.greater_than(Zflat, paddle.ones_like(Zflat) * MIN_FLOAT))
        out = paddle.reshape(Zflat, shape=[x.shape[0], -1])  # (batch_size, -1)

        for idx, layer in enumerate(self.layers):
            out = layer(out)

            if idx != len(self.layers) - 1:
                out = F.relu(out)

        return F.sigmoid(out)
