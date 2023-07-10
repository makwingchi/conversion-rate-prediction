import paddle
import paddle.nn as nn

from .dcnv3_helper import DeepAndCrossV3


class ShareBottom(nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.sparse_feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"]["common"]["sparse_feature_dim"]

        self.num_tasks = config["models"]["sharebottom"]["num_tasks"]

        self.embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.sparse_feature_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        self.dcn = []
        for i in range(self.num_tasks):
            curr_dcn = DeepAndCrossV3(config)
            self.dcn.append(curr_dcn)

    def forward(self, features, mask):
        outputs = []

        for i in range(self.num_tasks):
            curr_dcn = self.dcn[i]
            curr_out = curr_dcn(features, mask, self.embedding)

            outputs.append(curr_out)

        return outputs
