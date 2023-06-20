import math
import paddle
import numpy as np

from net import DNNLayer, StaticDNNLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.infer_target_var = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.is_distributed = False
        self.distributed_embedding = False

        if self.config.get("hyper_parameters.distributed_embedding", 0) == 1:
            self.distributed_embedding = True

        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.sparse_inputs_slots = self.config.get(
            "hyper_parameters.sparse_inputs_slots")
        self.dense_input_dim = self.config.get(
            "hyper_parameters.dense_input_dim")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.fc_sizes = self.config.get("hyper_parameters.fc_sizes")
    def create_feeds(self, is_infer=False):
        dense_input = paddle.static.data(
            name="dense_input",
            shape=[None, self.dense_input_dim],
            dtype="float32")

        sparse_input_ids = [
            paddle.static.data(
                name="C" + str(i), shape=[None, 1], lod_level=1, dtype="int64")
            for i in range(1, self.sparse_inputs_slots)
        ]

        label = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        feeds_list = [label] + sparse_input_ids
        return feeds_list

    def net(self, input, is_infer=False):
        self.log_key = input[0]
        self.label_input = input[1]
        self.sparse_inputs = input[2:self.sparse_inputs_slots]
        sparse_number = self.sparse_inputs_slots - 2

        def embedding_layer(input):
            if self.distributed_embedding:
                emb = fluid.contrib.layers.sparse_embedding(
                    input=input,
                    size=[
                        self.sparse_feature_number, self.sparse_feature_dim
                    ],
                    param_attr=fluid.ParamAttr(
                        name="SparseFeatFactors",
                        initializer=fluid.initializer.Uniform()))
            else:
                emb = paddle.fluid.layers.embedding(
                    input=input,
                    is_sparse=True,
                    is_distributed=self.is_distributed,
                    size=[
                        self.sparse_feature_number, self.sparse_feature_dim
                    ],
                    param_attr=paddle.fluid.ParamAttr(
                        name="SparseFeatFactors",
                        initializer=paddle.fluid.initializer.Uniform()))
            emb_sum = paddle.fluid.layers.sequence_pool(
                input=emb, pool_type='sum')
            return emb_sum

        sparse_embs = list(map(embedding_layer, self.sparse_inputs))

        dnn_model = StaticDNNLayer(
            self.sparse_feature_number, self.sparse_feature_dim,
            self.dense_input_dim, sparse_number, self.fc_sizes)

        pred = dnn_model.forward(sparse_embs)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)

        auc, batch_auc_var, _ = paddle.static.auc(input=predict_2d,
                                                  label=self.label_input,
                                                  slide_steps=0)
        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'log_key': self.log_key,'pred': pred}
            return fetch_dict

        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                self.label_input, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        self._cost = avg_cost
        fetch_dict = {'cost': avg_cost, 'auc': auc}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, lazy_mode=True)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
