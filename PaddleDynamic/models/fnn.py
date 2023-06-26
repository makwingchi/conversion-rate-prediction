import paddle
import paddle.nn.functional as F


class FNN(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["fnn"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["fnn"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["fnn"]["num_fields"]
        self.layer_sizes = self.config["models"]["fnn"]["fc_sizes"]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        sizes = [self.sparse_feature_dim * self.num_field] + self.layer_sizes + [1]

        self.nets = []

        for i in range(len(self.layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()
                ),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)
                )
            )

            self.nets.append(linear)

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature),
                axis=1
            )

            feature_ls.append(emb)

        x = paddle.concat(feature_ls, axis=1).astype("float32")

        for idx, net in enumerate(self.nets):
            x = net(x)

            if idx != len(self.nets) - 1:
                x = F.relu(x)

        return F.sigmoid(x)
