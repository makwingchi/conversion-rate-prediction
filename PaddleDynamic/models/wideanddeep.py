import paddle
import paddle.nn.functional as F


class WideAndDeep(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["wideanddeep"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["wideanddeep"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["wideanddeep"]["num_fields"]
        self.layer_sizes = self.config["models"]["wideanddeep"]["fc_sizes"]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        sizes = [self.sparse_feature_dim * self.num_field] + self.layer_sizes + [1]

        self.wide_model = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field,
            out_features=1,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0))
        )

        self.deep_models = []
        for i in range(len(sizes) - 1):
            deep_model = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0))
            )

            self.deep_models.append(deep_model)

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature),
                axis=1
            )

            feature_ls.append(emb)

        x = paddle.concat(feature_ls, axis=1).astype("float32")

        wide_out = self.wide_model(x)
        deep_out = x

        for idx, layer in enumerate(self.deep_models):
            deep_out = layer(deep_out)

            if idx != len(self.deep_models) - 1:
                deep_out = F.relu(deep_out)

        return F.sigmoid(wide_out + deep_out)
