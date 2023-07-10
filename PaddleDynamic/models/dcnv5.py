import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from .dcnv2 import DeepCrossLayer
from .deepcrossing import DeepCrossingResidualBlock


class SqueezeExcitation(nn.Layer):
    def __init__(self, num_fields, reduction_ratio=3, excitation_activation="ReLU"):
        super().__init__()

        self.num_fields = num_fields
        self.act = excitation_activation

        reduced_size = max(1, int(num_fields / reduction_ratio))
        excitation = [
            nn.Linear(
                num_fields,
                reduced_size,
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()
                ),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)
                )
            ),
            nn.ReLU(),
            nn.Linear(
                reduced_size,
                num_fields,
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()
                ),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)
                )
            )
        ]

        self.excitation = nn.Sequential(*excitation)

    def forward(self, feature_emb, mask):
        Z = paddle.mean(feature_emb, axis=-1)
        A = self.excitation(Z).unsqueeze(-1)

        weight = mask.unsqueeze(-1) + A

        if self.act.lower() == "relu":
            weight = F.relu(weight)
        elif self.act.lower() == "sigmoid":
            weight = F.sigmoid(weight)
        elif self.act.lower() == "softmax":
            weight = F.softmax(weight, axis=1)

        V = feature_emb * weight

        return V


class DeepAndCrossV5(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.num_fields = self.config["models"]["common"]["num_fields"]
        self.layer_sizes = self.config["models"]["common"]["fc_sizes"]
        self.activation_type = self.config["models"]["common"]["activate"]

        self.num_crosses = self.config["models"]["deepandcrossv2"]["num_crosses"]
        self.is_Stacked = self.config["models"]["deepandcrossv2"]["is_Stacked"]
        self.use_low_rank_mixture = self.config["models"]["deepandcrossv2"]["use_low_rank_mixture"]
        self.low_rank = self.config["models"]["deepandcrossv2"]["low_rank"]
        self.num_experts = 1
        self.init_value_ = 0.1

        self.multi_value = [9, 16, 17, 19, 20, 21]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        in_dim = self.sparse_feature_dim * self.num_fields

        self.DeepCrossLayer_ = DeepCrossLayer(
            in_dim,
            self.num_crosses,
            self.use_low_rank_mixture,
            self.low_rank,
            self.num_experts
        )

        self.residual_blocks = []

        for i in range(len(self.layer_sizes)):
            residual_block = DeepCrossingResidualBlock(
                input_dim=in_dim,
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
                in_features=in_dim * 2,
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(in_dim * 2))
                )
            )

        self.senet = []
        for _ in self.multi_value:
            self.senet.append(SqueezeExcitation(num_fields=20, excitation_activation="sigmoid"))

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            if idx in self.multi_value:
                continue

            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.embedding(feature),
                axis=1
            )

            feature_ls.append(emb.astype("float32"))

        for idx, multi in enumerate(self.multi_value):
            curr_senet = self.senet[idx]
            multi_feature = features[multi]
            multi_embedding = self.embedding(multi_feature)

            emb = paddle.sum(
                curr_senet(multi_embedding, mask[:, multi, :]),
                axis=1
            )

            feature_ls.append(emb.astype("float32"))

        x = paddle.concat(feature_ls, axis=1)

        # Model Structural: Stacked or Parallel
        if self.is_Stacked:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(x)
            # MLPLayer
            dnn_output = self.DNN_(cross_out)
            logit = self.fc(dnn_output)
            predict = F.sigmoid(logit)
        else:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(x)

            # Residual blocks
            dnn_output = x.clone()

            for block in self.residual_blocks:
                dnn_output = block(dnn_output)

            last_out = paddle.concat([dnn_output, cross_out], axis=-1)
            logit = self.fc(last_out)
            predict = F.sigmoid(logit)

        return predict
