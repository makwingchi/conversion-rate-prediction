import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from .dcnv2 import DeepCrossLayer
from .deepcrossing import DeepCrossingResidualBlock


class DeepAndCrossV9(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.num_fields = self.config["models"]["common"]["num_fields"]
        self.layer_sizes = self.config["models"]["common"]["fc_sizes"]
        self.activation_type = self.config["models"]["common"]["activate"]

        self.mlp_sizes = self.config["models"]["deepandcross"]["mlp_sizes"]
        self.num_crosses = self.config["models"]["deepandcrossv2"]["num_crosses"]
        self.is_Stacked = self.config["models"]["deepandcrossv2"]["is_Stacked"]
        self.use_low_rank_mixture = self.config["models"]["deepandcrossv2"]["use_low_rank_mixture"]
        self.low_rank = self.config["models"]["deepandcrossv2"]["low_rank"]
        self.num_experts = 1
        self.init_value_ = 0.1

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        self.DeepCrossLayer_ = DeepCrossLayer(
            self.num_fields * self.sparse_feature_dim,
            self.num_crosses,
            self.use_low_rank_mixture,
            self.low_rank,
            self.num_experts
        )

        self.residual_blocks = []
        for i in range(len(self.layer_sizes)):
            residual_block = DeepCrossingResidualBlock(
                input_dim=self.sparse_feature_dim * self.num_fields,
                hidden_dim=self.layer_sizes[i],
                act=self.activation_type
            )

            self.residual_blocks.append(residual_block)

        self.final_residual = []
        for i in range(len(self.layer_sizes)):
            residual_block = DeepCrossingResidualBlock(
                input_dim=self.sparse_feature_dim * self.num_fields * 2,
                hidden_dim=self.layer_sizes[i],
                act=self.activation_type
            )

            self.final_residual.append(residual_block)

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
                in_features=self.num_fields * self.sparse_feature_dim * 2,
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(self.num_fields * self.sparse_feature_dim * 2))
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

        feat_embeddings = paddle.reshape(x, shape=[-1, self.num_fields * self.sparse_feature_dim])

        # Model Structural: Stacked or Parallel
        if self.is_Stacked:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(feat_embeddings)
            # MLPLayer
            dnn_output = self.DNN_(cross_out)
            logit = self.fc(dnn_output)
            predict = F.sigmoid(logit)
        else:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(feat_embeddings)

            # Residual blocks
            dnn_output = feat_embeddings.clone()

            for block in self.residual_blocks:
                dnn_output = block(dnn_output)

            last_out = paddle.concat([dnn_output, cross_out], axis=-1)
            # residual blocks
            for block in self.final_residual:
                last_out = block(last_out)

            logit = self.fc(last_out)
            predict = F.sigmoid(logit)

        return predict
