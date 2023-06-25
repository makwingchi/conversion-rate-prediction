import paddle
import paddle.nn.functional as F


class DeepCrossingResidualBlock(paddle.nn.Layer):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.linear1 = paddle.nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )
        )
        self.linear2 = paddle.nn.Linear(
            in_features=hidden_dim,
            out_features=input_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )
        )

    def forward(self, x):
        out1 = F.relu(self.linear1(x))
        out2 = self.linear2(out1) + x

        return F.relu(out2)


class DeepCrossing(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["deepcrossing"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["deepcrossing"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["deepcrossing"]["num_fields"]
        self.layer_sizes = self.config["models"]["deepcrossing"]["fc_sizes"]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        self.residual_blocks = []

        for i in range(len(self.layer_sizes)):
            residual_block = DeepCrossingResidualBlock(
                input_dim=self.sparse_feature_dim * self.num_field,
                hidden_dim=self.layer_sizes[i]
            )

            self.residual_blocks.append(residual_block)

        self.final_linear = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field,
            out_features=1,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )
        )

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature),
                axis=1
            )

            feature_ls.append(emb)

        x = paddle.concat(feature_ls, axis=1).astype("float32")

        for block in self.residual_blocks:
            x = block(x)

        out = self.final_linear(x)

        return F.sigmoid(out)
