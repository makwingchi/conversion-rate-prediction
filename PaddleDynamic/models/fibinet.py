import numpy as np
import paddle
import paddle.nn as nn


class SqueezeExcitation(nn.Layer):
    def __init__(self, num_fields, reduction_ratio=3, excitation_activation="ReLU"):
        super().__init__()

        self.num_fields = num_fields

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

        if excitation_activation.lower() == "relu":
            excitation.append(nn.ReLU())
        elif excitation_activation.lower() == "sigmoid":
            excitation.append(nn.Sigmoid())
        elif excitation_activation.lower() == "softmax":
            excitation.append(nn.Softmax())
        else:
            raise NotImplementedError

        self.excitation = nn.Sequential(*excitation)

    def forward(self, feature_emb):
        Z = paddle.mean(feature_emb, axis=-1)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)

        return V


class BilinearInteractionV2(nn.Layer):
    def __init__(self, num_fields, embedding_dim, bilinear_type="field_interaction"):
        super().__init__()

        self.bilinear_type = bilinear_type
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.interact_dim = int(num_fields * (num_fields - 1) / 2)

        if self.bilinear_type == "field_all":
            self.bilinear_W = paddle.create_parameter((embedding_dim, embedding_dim), dtype="float32")
        elif self.bilinear_type == "field_each":
            self.bilinear_W = paddle.create_parameter((num_fields, embedding_dim, embedding_dim), dtype="float32")
        elif self.bilinear_type == "field_interaction":
            self.bilinear_W = paddle.create_parameter((self.interact_dim, embedding_dim, embedding_dim),
                                                      dtype="float32")
        else:
            raise NotImplementedError

        indices = np.array(np.tril_indices(num_fields, k=-1, m=num_fields))
        w_attr = paddle.ParamAttr(initializer=nn.initializer.Assign(value=indices))

        self.triu_index = paddle.create_parameter(shape=indices.shape, dtype='int32', attr=w_attr)
        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierNormal(self.bilinear_W)

    def forward(self, feature_emb):
        if self.bilinear_type == "field_interaction":
            left_emb = paddle.index_select(feature_emb, self.triu_index[0], 1)
            right_emb = paddle.index_select(feature_emb, self.triu_index[1], 1)
            bilinear_out = paddle.matmul(left_emb.unsqueeze(-1), self.bilinear_W).squeeze(-1) * right_emb
        else:
            if self.bilinear_type == "field_all":
                hidden_emb = paddle.matmul(feature_emb, self.bilinear_W)
            elif self.bilinear_type == "field_each":
                hidden_emb = paddle.matmul(feature_emb.unsqueeze(-1), self.bilinear_W).squeeze(2)

            left_emb = paddle.index_select(hidden_emb,  self.triu_index[0], 1)
            right_emb = paddle.index_select(feature_emb, self.triu_index[1], 1)
            bilinear_out = left_emb * right_emb

        return bilinear_out


class FiBiNet(nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.sparse_feature_dim = self.config["models"]["fibinet"]["sparse_feature_dim"]
        self.num_fields = self.config["models"]["fibinet"]["sparse_inputs_slots"] - 2

        sqz_reduction_ratio = self.config["models"]["fibinet"]["reduction_ratio"]
        bilinear_type = self.config["models"]["fibinet"]["bilinear_type"].lower()
        hidden_units = self.config["models"]["fibinet"]["hidden_units"]
        hidden_activations = self.config["models"]["fibinet"]["hidden_activations"]

        # feature importance extraction
        self.senet_layer = SqueezeExcitation(self.num_fields, sqz_reduction_ratio, "ReLU")

        # interaction 1 for orig embedding result
        self.bilinear_interaction1 = BilinearInteractionV2(self.num_fields, self.sparse_feature_dim, bilinear_type)

        # interaction 2 for senet result
        self.bilinear_interaction2 = BilinearInteractionV2(self.num_fields, self.sparse_feature_dim, bilinear_type)

        # dnn layer with no dropout currently
        # todo: add dropout
        dnn_net = []
        input_dim = self.num_fields * (self.num_fields - 1) * self.sparse_feature_dim
        hidden_units = [input_dim] + hidden_units

        for idx in range(len(hidden_units) - 1):
            dnn_net.append(
                nn.Linear(
                    hidden_units[idx],
                    hidden_units[idx + 1],
                    weight_attr=paddle.framework.ParamAttr(
                        initializer=paddle.nn.initializer.XavierUniform()
                    ),
                    bias_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.0)
                    )
                )
            )

            if hidden_activations is not None:
                if hidden_activations == 'ReLU':
                    dnn_net.append(nn.ReLU())
                else:
                    raise NotImplementedError

        dnn_net.append(
            nn.Linear(
                hidden_units[-1],
                1,
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()
                ),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)
                )
            )
        )

        self.dnn = nn.Sequential(*dnn_net)

    def forward(self, sparse_embs):
        x = paddle.concat(sparse_embs, axis=1)
        feat_len = x.shape[-1]
        x = x.reshape((-1, self.num_fields, feat_len//self.num_fields))

        senet_emb = self.senet_layer(x)
        bilinear_p = self.bilinear_interaction1(senet_emb)
        bilinear_q = self.bilinear_interaction2(x)

        comb_out = paddle.flatten(paddle.concat([bilinear_p, bilinear_q], axis=-1), start_axis=1)

        dnn_out = self.dnn(comb_out)
        m = nn.Sigmoid()

        return m(dnn_out).reshape((-1,1))
