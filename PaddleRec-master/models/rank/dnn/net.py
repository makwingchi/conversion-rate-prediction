import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class DNNLayer(nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 sparse_feature_dim,
                 dense_feature_dim,
                 num_field,
                 layer_sizes,
                 sync_mode=None):
        super(DNNLayer, self).__init__()
        self.sync_mode = sync_mode
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        use_sparse = True
        if paddle.is_compiled_with_npu():
            use_sparse = False

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        sizes = [sparse_feature_dim * num_field + dense_feature_dim
                 ] + self.layer_sizes + [2]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, sparse_inputs, dense_inputs, show_click=None):

        sparse_embs = []
        for s_input in sparse_inputs:
            if self.sync_mode == "gpubox":
                emb = paddle.fluid.contrib.sparse_embedding(
                    input=s_input,
                    size=[
                        self.sparse_feature_number, self.sparse_feature_dim + 2
                    ],
                    param_attr=paddle.ParamAttr(name="embedding"))
                emb = paddle.fluid.layers.continuous_value_model(
                    emb, show_click, False)
            else:
                emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            sparse_embs.append(emb)

        y_dnn = paddle.concat(x=sparse_embs + [dense_inputs], axis=1)

        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)

        return y_dnn


class StaticDNNLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(StaticDNNLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        sizes = [sparse_feature_dim * num_field 
                 ] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, sparse_embs):
        y_dnn = paddle.concat(x=sparse_embs, axis=1)
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)

        predict = F.sigmoid(y_dnn)
        return predict
