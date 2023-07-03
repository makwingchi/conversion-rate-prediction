import paddle
import paddle.nn.functional as F


class DeepFM(paddle.nn.Layer):
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

        sizes = [self.sparse_feature_dim * self.num_field] + self.layer_sizes + [1]

        self.layers = []
        for i in range(len(sizes)-1):
            layer = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i+1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0))
            )

            self.layers.append(layer)

        self.linear = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field,
            out_features=1,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0))
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

        sparse_emb = paddle.concat(feature_ls, axis=1).astype("float32")  # batch x 26 x emb_dim

        flatten_x = paddle.reshape(sparse_emb, shape=[-1, self.num_field * self.sparse_feature_dim])
        y_first_order = self.linear(flatten_x)  # batch x 1

        # sum of square
        sum_feature_emb = paddle.sum(sparse_emb, axis=1)  # batch x emb_dim
        sum_feature_emb_square = paddle.square(sum_feature_emb)

        # square of sum
        square_feature_emb = paddle.square(sparse_emb)
        square_feature_emb_sum = paddle.sum(square_feature_emb, axis=1)  # batch x emb_dim

        y_second_order = 0.5 * paddle.sum(sum_feature_emb_square - square_feature_emb_sum, axis=1, keepdim=True)  # batch x 1

        y_dnn = paddle.reshape(sparse_emb, shape=[-1, self.num_field * self.sparse_feature_dim])

        for idx, layer in enumerate(self.layers):
            y_dnn = layer(y_dnn)

            if idx != len(self.layers) - 1:
                y_dnn = F.relu(y_dnn)

        return F.sigmoid(y_first_order + y_second_order + y_dnn)
