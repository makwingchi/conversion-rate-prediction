import paddle
import paddle.nn.functional as F

from .deepcrossing import DeepCrossingResidualBlock


class DeepCrossingAttn(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["common"]["num_fields"]
        self.layer_sizes = self.config["models"]["common"]["fc_sizes"]
        self.activation_type = self.config["models"]["common"]["activate"]
        self.attention_sizes = self.config["models"]["deepcrossingattn"]["attention_sizes"]

        self.user_pairs = []
        self.ad_pairs = []
        self.__assign_pairs()

        self.pairs = self.user_pairs + self.ad_pairs

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
                input_dim=self.sparse_feature_dim * (self.num_field - len(self.user_multi_value) - len(self.ad_multi_value) + len(self.pairs)),
                hidden_dim=self.layer_sizes[i],
                act=self.activation_type
            )

            self.residual_blocks.append(residual_block)

        self.final_linear = paddle.nn.Linear(
            in_features=self.sparse_feature_dim * (self.num_field - len(self.user_multi_value) - len(self.ad_multi_value) + len(self.pairs)),
            out_features=1,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )
        )

        self.attention_layers = []
        for i in range(len(self.pairs)):
            curr_attention = []
            sizes = [self.sparse_feature_dim * 4] + self.attention_sizes + [1]

            for j in range(len(sizes) - 1):
                linear = paddle.nn.Linear(
                    in_features=sizes[j],
                    out_features=sizes[j + 1],
                    weight_attr=paddle.framework.ParamAttr(
                        initializer=paddle.nn.initializer.XavierUniform()
                    ),
                    bias_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.0)
                    )
                )

                curr_attention.append(linear)

                if j != len(sizes) - 2:
                    act = paddle.nn.ReLU()
                    curr_attention.append(act)

            self.attention_layers.append(paddle.nn.Sequential(*curr_attention))

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            if idx in self.user_multi_value or idx in self.ad_multi_value:
                continue

            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature),
                axis=1
            )

            feature_ls.append(emb)

        for idx, (multi, single) in enumerate(self.pairs):
            single_feature = features[single]
            multi_feature = features[multi]
            curr_mask = mask[:, multi, :].unsqueeze(-1)

            expanded_single = paddle.expand(
                paddle.max(single_feature, axis=-1).unsqueeze(-1),
                shape=[-1, single_feature.shape[1]]
            )

            single_embedding = self.embedding(expanded_single)
            multi_embedding = self.embedding(multi_feature)

            concat = paddle.concat(
                [
                    multi_embedding, single_embedding, multi_embedding - single_embedding, multi_embedding * single_embedding
                ],
                axis=-1
            )

            concat = self.attention_layers[idx](concat)
            attn_fc = concat + curr_mask
            attn_fc = attn_fc.transpose(perm=(0, 2, 1))
            weight = F.softmax(attn_fc)
            weighted_pooling = paddle.matmul(weight, multi_embedding)

            feature_ls.append(weighted_pooling)

        x = paddle.concat(feature_ls, axis=1).astype("float32")

        for block in self.residual_blocks:
            x = block(x)

        out = self.final_linear(x)

        return F.sigmoid(out)

    def __assign_pairs(self):
        self.user_single_value = [i - 1 for i in range(1, 10)] + [10, 12]
        self.user_multi_value = [9]

        self.ad_single_value = [18, 22, 23]
        self.ad_multi_value = [16, 17, 19, 20, 21]

        for single_value in self.user_single_value:
            for multi_value in self.user_multi_value:
                self.user_pairs.append([multi_value, single_value])

        for single_value in self.ad_single_value:
            for multi_value in self.ad_multi_value:
                self.ad_pairs.append([multi_value, single_value])
