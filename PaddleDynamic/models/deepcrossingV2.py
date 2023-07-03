import paddle
import paddle.nn.functional as F

from .deepcrossing import DeepCrossingResidualBlock


class DeepCrossingV2(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.layer_sizes = self.config["models"]["common"]["fc_sizes"]
        self.emb_size_map = self.__get_emb_size_map()

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
                input_dim=sum([min(i, self.sparse_feature_dim) for i in list(self.emb_size_map.values())]),
                hidden_dim=self.layer_sizes[i]
            )

            self.residual_blocks.append(residual_block)

        self.final_linear = paddle.nn.Linear(
            in_features=sum([min(i, self.sparse_feature_dim) for i in list(self.emb_size_map.values())]),
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

            end = self.emb_size_map[idx]
            feature_ls.append(emb[:, :end])

        x = paddle.concat(feature_ls, axis=1).astype("float32")

        for block in self.residual_blocks:
            x = block(x)

        out = self.final_linear(x)

        return F.sigmoid(out)

    def __get_emb_size_map(self):
        emb_size_map = {
            0: 2, 1: 2, 2: 41, 3: 38, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 41,
            10: 3, 11: 2, 12: 3, 13: 2, 14: 5, 15: 2, 16: 54, 17: 37,
            18: 21, 19: 12, 20: 2, 21: 64, 22: 19, 23: 5, 24: 5, 25: 7
        }

        return emb_size_map
