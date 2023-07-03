import paddle
import paddle.nn.functional as F


class NFM(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["common"]["num_fields"]
        self.layer_sizes = self.config["models"]["common"]["fc_sizes"]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        sizes = [self.sparse_feature_dim] + self.layer_sizes + [1]

        self.linear = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field,
            out_features=1,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0))
        )

        self.layers = []
        for i in range(len(sizes) - 1):
            layer = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0))
            )

            self.layers.append(layer)

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

        sum_of_square = paddle.sum(x, axis=1) ** 2
        square_of_sum = paddle.sum(x ** 2, axis=1)

        bi_interaction = (sum_of_square - square_of_sum) * 0.5

        flatten_x = paddle.reshape(x, shape=[-1, self.num_field * self.sparse_feature_dim])
        linear_out = self.linear(flatten_x)

        for idx, layer in enumerate(self.layers):
            bi_interaction = layer(bi_interaction)

            if idx != len(self.layers) - 1:
                bi_interaction = F.relu(bi_interaction)

        return F.sigmoid(bi_interaction + linear_out)
