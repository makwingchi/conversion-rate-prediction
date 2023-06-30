import paddle
import paddle.nn.functional as F

from .deepcrossing import DeepCrossingResidualBlock


class NativeAttention(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["common"]["num_fields"]
        self.layer_sizes = self.config["models"]["common"]["fc_sizes"]
        self.qkv_dim = self.config["models"]["native_attention"]["qkv_dim"]

        self.single_value_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 19, 23, 24]
        self.single_value_cols = [i - 1 for i in self.single_value_cols]

        self.WQ = paddle.nn.Linear(
            in_features=self.sparse_feature_dim,
            out_features=self.qkv_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            ),
            bias_attr=False
        )

        self.WK = paddle.nn.Linear(
            in_features=self.sparse_feature_dim,
            out_features=self.qkv_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            ),
            bias_attr=False
        )

        self.WV = paddle.nn.Linear(
            in_features=self.sparse_feature_dim,
            out_features=self.qkv_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            ),
            bias_attr=False
        )

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
                input_dim=self.sparse_feature_dim * self.num_field + (self.num_field * self.num_field - len(self.single_value_cols)) * self.qkv_dim,
                hidden_dim=self.layer_sizes[i]
            )

            self.residual_blocks.append(residual_block)

        self.final_linear = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * self.num_field + (self.num_field * self.num_field - len(self.single_value_cols)) * self.qkv_dim,
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
        raw_ls = []

        # sum pooling
        for idx, feature in enumerate(features):
            masked_emb = paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature)
            emb = paddle.sum(masked_emb, axis=1)

            raw_ls.append(masked_emb.astype("float32"))
            feature_ls.append(emb.astype("float32"))

        # attention
        raw_features = paddle.concat(raw_ls, axis=1)

        Q = self.WQ(raw_features)
        K = self.WK(raw_features)
        V_ls = [self.WV(raw_feature) for raw_feature in raw_ls]

        attention = Q @ K.transpose(perm=(0, 2, 1))
        attention_ls = paddle.split(attention, num_or_sections=self.num_field, axis=1)

        for idx1, att in enumerate(attention_ls):
            curr_att = paddle.split(att, num_or_sections=self.num_field, axis=-1)
            curr_V = V_ls[idx1]

            for idx2, ca in enumerate(curr_att):
                # self-attention is unnecessary for single value columns
                if idx2 in self.single_value_cols and idx2 == idx1:
                    continue

                feature_ls.append(
                    paddle.sum(F.softmax(ca / self.qkv_dim ** 0.5) @ curr_V, axis=1)
                )

        # for i in range(len(features)):
        #     if i not in [9, 16, 17, 19, 20, 21, 24, 25]:
        #         continue
        #
        #     f1 = paddle.exp(mask[:, i, :].unsqueeze(-1)) * self.embedding(features[i])
        #     Q = self.WQ(f1.astype("float32"))
        #     V = self.WV(f1.astype("float32"))
        #
        #     for j in range(len(features)):
        #         if j not in [9, 16, 17, 19, 20, 21, 24, 25]:
        #             continue
        #
        #         f2 = paddle.exp(mask[:, j, :].unsqueeze(-1)) * self.embedding(features[j])
        #         K = self.WK(f2.astype("float32"))
        #
        #         attention = F.softmax((Q @ K.transpose(perm=(0, 2, 1))) / self.qkv_dim ** 0.5)
        #         feature = paddle.sum(attention @ V, axis=1)
        #
        #         feature_ls.append(feature)

        x = paddle.concat(feature_ls, axis=1)

        for block in self.residual_blocks:
            x = block(x)

        out = self.final_linear(x)

        return F.sigmoid(out)
