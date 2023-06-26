import paddle
import paddle.nn.functional as F


class DeepAndCross(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["deepandcross"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["deepandcross"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["deepandcross"]["num_fields"]
        self.layer_sizes = self.config["models"]["deepandcross"]["fc_sizes"]
        self.num_crosses = self.config["models"]["deepandcross"]["num_crosses"]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        sizes = [self.sparse_feature_dim * self.num_field] + self.layer_sizes

        self.deep_models = []

        for i in range(len(self.layer_sizes)):
            deep_model = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()
                ),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)
                )
            )

            self.deep_models.append(deep_model)

        self.cross_models = []

        for i in range(self.num_crosses):
            cross_model = paddle.nn.Linear(
                in_features=self.sparse_feature_dim * self.num_field,
                out_features=1,
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()
                ),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)
                )
            )

            self.cross_models.append(cross_model)

        self.final_linear = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field + sizes[-1],
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

        # cross
        x0 = x.unsqueeze(-1)
        xl = x0

        for cross_model in self.cross_models:
            xl_T = paddle.transpose(xl, perm=(0, 2, 1))
            xl = cross_model(paddle.matmul(x0, xl_T)) + xl

        cross_out = xl.squeeze(-1)

        # deep
        deep_out = x
        for idx, deep_model in enumerate(self.deep_models):
            deep_out = deep_model(deep_out)

            if idx != len(self.deep_models) - 1:
                deep_out = F.relu(deep_out)

        concat_out = paddle.concat([cross_out, deep_out], axis=-1)

        return F.sigmoid(self.final_linear(concat_out))
