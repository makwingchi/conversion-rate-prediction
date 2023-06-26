import paddle
import paddle.nn.functional as F


class PNN(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["pnn"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["pnn"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["pnn"]["num_fields"]
        self.layer_sizes = self.config["models"]["pnn"]["fc_sizes"]
        self.num_matrices = self.config["models"]["pnn"]["num_matrices"]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        sizes = [self.num_matrices] + self.layer_sizes + [1]

        self.wz = []
        self.wp = []

        for i in range(self.num_matrices):
            curr_wz = paddle.create_parameter(
                shape=[self.sparse_feature_dim, self.num_field],
                dtype="float32"
            )

            curr_wp = paddle.create_parameter(
                shape=[self.num_field, self.num_field],
                dtype="float32"
            )

            self.wz.append(curr_wz)
            self.wp.append(curr_wp)

        self.bias = paddle.create_parameter(shape=[1, self.num_matrices], dtype="float32")

        self.linear_layers = []

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

            self.linear_layers.append(linear)

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature),
                axis=1
            )

            feature_ls.append(emb)

        x = paddle.concat(feature_ls, axis=1).astype("float32")

        f_prime = x.reshape(shape=[-1, self.num_field, self.sparse_feature_dim])
        f = f_prime.transpose(perm=(0, 2, 1))

        p = paddle.matmul(f_prime, f)

        lz_ls, lp_ls = [], []

        for wz in self.wz:
            curr_lz = paddle.sum(
                paddle.sum(paddle.multiply(wz, f), axis=1),
                axis=-1,
                keepdim=True
            )

            lz_ls.append(curr_lz)

        for wp in self.wp:
            curr_lp = paddle.sum(
                paddle.sum(paddle.multiply(wp, p), axis=1),
                axis=-1,
                keepdim=True
            )

            lp_ls.append(curr_lp)

        lz = paddle.concat(lz_ls, axis=-1)
        lp = paddle.concat(lp_ls, axis=-1)

        out = lz + lp + self.bias

        for idx, layer in enumerate(self.linear_layers):
            out = layer(out)

            if idx != len(self.linear_layers) - 1:
                out = F.relu(out)

        return F.sigmoid(out)
