import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class InteractingLayer(nn.Layer):
    def __init__(self, embedding_size, interact_layer_size, head_num,
                 use_residual, scaling):
        super().__init__()

        self.attn_emb_size = interact_layer_size // head_num
        self.head_num = head_num
        self.use_residual = use_residual
        self.scaling = scaling

        self.W_Query = (
            paddle.create_parameter(
                shape=[embedding_size, interact_layer_size],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierUniform()
            )
        )
        self.W_Key = (
            paddle.create_parameter(
                shape=[embedding_size, interact_layer_size],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierUniform()
            )
        )
        self.W_Value = (
            paddle.create_parameter(
                shape=[embedding_size, interact_layer_size],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        if self.use_residual:
            self.W_Res = (
                paddle.create_parameter(
                    shape=[embedding_size, interact_layer_size],
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.XavierUniform()
                )
            )

        self.layer_norm = paddle.nn.LayerNorm(interact_layer_size, epsilon=1e-08)

    def forward(self, inputs):
        querys = F.relu(inputs @ self.W_Query)
        keys = F.relu(inputs @ self.W_Key)
        values = F.relu(inputs @ self.W_Value)

        q = paddle.stack(paddle.split(querys, self.head_num, axis=2))
        k = paddle.stack(paddle.split(keys, self.head_num, axis=2))
        v = paddle.stack(paddle.split(values, self.head_num, axis=2))

        inner_prod = q @ k.transpose(perm=(0, 1, 3, 2))

        if self.scaling:
            inner_prod /= self.attn_emb_size ** 0.5

        normalized_attn_scores = F.softmax(inner_prod, axis=-1)
        result = paddle.matmul(normalized_attn_scores, v)

        result = paddle.concat(paddle.split(result, self.head_num), axis=-1).squeeze(axis=0)

        if self.use_residual:
            result += F.relu(inputs @ self.W_Res)

        result = F.relu(result)
        result = self.layer_norm(result)

        return result


class EmbeddingLayer(nn.Layer):
    def __init__(self, feature_number, embedding_dim, num_field, fc_sizes):
        super().__init__()

        self.feature_number = feature_number
        self.embedding_dim = embedding_dim
        self.num_field = num_field
        self.fc_sizes = fc_sizes

        self.feature_embeddings = paddle.nn.Embedding(
            self.feature_number,
            self.embedding_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(0, 0.01)
            )
        )

        if len(self.fc_sizes) > 0:
            self.dnn_layers = []
            linear = paddle.nn.Linear(
                in_features=num_field * embedding_dim,
                out_features=fc_sizes[0],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        0,
                        math.sqrt(2. /
                                  (num_field * embedding_dim + fc_sizes[0])))),
                bias_attr=paddle.nn.initializer.Normal(
                    0,
                    math.sqrt(2. / (num_field * embedding_dim + fc_sizes[0]))))
            self.add_sublayer('linear_0', linear)
            self.add_sublayer('relu_0', paddle.nn.ReLU())
            self.dnn_layers.append(linear)
            for i in range(1, len(fc_sizes)):
                linear = paddle.nn.Linear(
                    in_features=fc_sizes[i - 1],
                    out_features=fc_sizes[i],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Normal(
                            0, math.sqrt(2. /
                                         (fc_sizes[i - 1] + fc_sizes[i])))),
                    bias_attr=paddle.nn.initializer.Normal(
                        0, math.sqrt(2. / (fc_sizes[i - 1] + fc_sizes[i]))))
                self.add_sublayer('linear_%d' % i, linear)
                self.dnn_layers.append(linear)
                norm = paddle.nn.BatchNorm(
                    fc_sizes[i],
                    is_test=self.training,
                    momentum=0.99,
                    epsilon=0.001)
                self.add_sublayer('norm_%d' % i, norm)
                self.dnn_layers.append(norm)
                act = paddle.nn.ReLU()
                self.add_sublayer('relu_%d' % i, act)
                self.dnn_layers.append(act)
                self.add_sublayer('dropout_%d' % i, paddle.nn.Dropout())
            linear = paddle.nn.Linear(
                in_features=fc_sizes[-1],
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        0, math.sqrt(2. / (fc_sizes[-1] + 1)))),
                bias_attr=paddle.nn.initializer.Normal())
            self.add_sublayer('pred_dense', linear)
            self.dnn_layers.append(linear)

    def forward(self, features, mask):
        feature_ls = []

        for idx, feature in enumerate(features):
            emb = paddle.sum(
                paddle.exp(mask[:, idx, :].unsqueeze(-1)) * self.feature_embeddings(feature),
                axis=1,
                keepdim=True
            )  # batch x 1 x emb_dim

            feature_ls.append(emb)

        emb = paddle.concat(feature_ls, axis=1).astype("float32")  # batch x 26 x emb_size

        y_first_order = None

        if len(self.fc_sizes) > 0:
            y_dense = emb.reshape(shape=[-1, self.num_field * self.embedding_dim])

            for layer in self.dnn_layers:
                y_dense = layer(y_dense)
        else:
            y_dense = None

        return emb, y_first_order, y_dense


class AutoInt(nn.Layer):
    def __init__(self, config, use_residual=True, scaling=True):
        super().__init__()

        self.config = config
        self.embedding_dim = self.config["models"]["common"]["sparse_feature_dim"]
        self.num_field = self.config["models"]["common"]["num_fields"]
        self.fc_sizes = self.config["models"]["common"]["fc_sizes"]
        self.feature_number = self.config["models"]["common"]["sparse_feature_number"]
        self.attn_layer_sizes = self.config["models"]["autoint"]["attn_layer_sizes"]
        self.head_num = self.config["models"]["autoint"]["head_num"]

        self.output_size = self.attn_layer_sizes[-1]

        self.emb_layer = EmbeddingLayer(
            self.feature_number, self.embedding_dim, self.num_field, self.fc_sizes
        )

        self.attn_layer_sizes = [self.embedding_dim] + self.attn_layer_sizes
        self.interaction_layers = nn.Sequential(*[
            InteractingLayer(
                self.attn_layer_sizes[i],
                self.attn_layer_sizes[i + 1], self.head_num,
                use_residual, scaling
            )
            for i in range(len(self.attn_layer_sizes) - 1)
        ])

        self.linear = nn.Linear(
            self.output_size * self.num_field,
            1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    0, math.sqrt(2. / (self.output_size * self.num_field + 1))
                )
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal()
            )
        )

    def forward(self, features, mask):
        out, y_first_order, y_dense = self.emb_layer(features, mask)

        for layer in self.interaction_layers:
            out = layer(out)

        out = paddle.flatten(out, start_axis=1)
        out = self.linear(out)

        if len(self.fc_sizes) > 0:
            out += y_dense

        return F.sigmoid(out)
