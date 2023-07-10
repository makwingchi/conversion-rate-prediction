import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from .dcnv2 import DeepCrossLayer
from .deepcrossing import DeepCrossingResidualBlock
from .fibinet import SqueezeExcitation


class DeepAndCrossV10(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.num_fields = self.config["models"]["common"]["num_fields"]
        self.layer_sizes = self.config["models"]["common"]["fc_sizes"]
        self.activation_type = self.config["models"]["common"]["activate"]

        self.attention_sizes = self.config["models"]["deepcrossingattn"]["attention_sizes"]
        self.num_crosses = self.config["models"]["deepandcrossv2"]["num_crosses"]
        self.is_Stacked = self.config["models"]["deepandcrossv2"]["is_Stacked"]
        self.use_low_rank_mixture = self.config["models"]["deepandcrossv2"]["use_low_rank_mixture"]
        self.low_rank = self.config["models"]["deepandcrossv2"]["low_rank"]
        self.excitation = self.config["models"]["deepandcrossv6"]["excitation"]

        self.num_experts = 1
        self.init_value_ = 0.1

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

        self.in_dim = self.sparse_feature_dim * (self.num_fields + len(self.pairs) - len(self.ad_multi_value) - len(self.user_multi_value))

        self.DeepCrossLayer_ = DeepCrossLayer(
            self.in_dim,
            self.num_crosses,
            self.use_low_rank_mixture,
            self.low_rank,
            self.num_experts
        )

        self.residual_blocks = []

        for i in range(len(self.layer_sizes)):
            residual_block = DeepCrossingResidualBlock(
                input_dim=self.in_dim,
                hidden_dim=self.layer_sizes[i],
                act=self.activation_type
            )

            self.residual_blocks.append(residual_block)

        if self.is_Stacked:
            self.fc = paddle.nn.Linear(
                in_features=self.layer_sizes[-1],
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(self.layer_sizes[-1]))
                )
            )
        else:
            self.fc = paddle.nn.Linear(
                in_features=self.in_dim * 2,
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(self.in_dim * 2))
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

        self.senet = SqueezeExcitation(num_fields=self.in_dim//self.sparse_feature_dim, excitation_activation=self.excitation)

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            if idx in self.ad_multi_value or idx in self.user_multi_value:
                continue

            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature),
                axis=1
            )

            feature_ls.append(emb.astype("float32"))

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

            feature_ls.append(weighted_pooling.squeeze(1))

        x = paddle.concat(feature_ls, axis=1)
        se_out = self.senet(x.reshape(shape=[-1, self.in_dim//self.sparse_feature_dim, self.sparse_feature_dim]))
        se_out = se_out.reshape(shape=[-1, self.in_dim])

        # Model Structural: Stacked or Parallel
        if self.is_Stacked:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(se_out)
            # MLPLayer
            dnn_output = self.DNN_(cross_out)
            logit = self.fc(dnn_output)
            predict = F.sigmoid(logit)
        else:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(se_out)

            # Residual blocks
            dnn_output = se_out.clone()

            for block in self.residual_blocks:
                dnn_output = block(dnn_output)

            last_out = paddle.concat([dnn_output, cross_out], axis=-1)
            logit = self.fc(last_out)
            predict = F.sigmoid(logit)

        return predict

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
