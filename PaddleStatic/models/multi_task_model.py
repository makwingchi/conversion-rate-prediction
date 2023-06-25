import paddle

from .mmoe import MMoE


class StaticMultiTaskModel:
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()
        self._get_model()

    def _init_hyper_parameters(self):
        self.model_type = self.config["runner"]["model_type"].lower()
        self.sparse_inputs_slots = self.config["models"][self.model_type]["sparse_inputs_slots"]
        self.learning_rate = self.config["optimizer"]["learning_rate"]
        self.sparse_feature_number = self.config["models"][self.model_type]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"][self.model_type]["sparse_feature_dim"]

    def _get_model(self):
        _map = {
            "mmoe": MMoE
        }

        self.model = _map[self.model_type]

    def create_feeds(self):
        sparse_input_ids = [
            paddle.static.data(
                name="C" + str(i), shape=[None, 1], lod_level=1, dtype="int64")
            for i in range(1, self.sparse_inputs_slots)
        ]

        t1 = paddle.static.data(
            name="t1", shape=[None, 1], dtype="int64"
        )

        t2 = paddle.static.data(
            name="t2", shape=[None, 1], dtype="int64"
        )

        t3 = paddle.static.data(
            name="t3", shape=[None, 1], dtype="int64"
        )

        return [t1, t2, t3] + sparse_input_ids

    def net(self, _input, is_infer=False):
        self.log_key = _input[0]
        self.t1 = _input[1]
        self.t2 = _input[2]
        self.t3 = _input[3]
        self.sparse_inputs = _input[4:]

        def embedding_layer(_input):
            emb = paddle.fluid.layers.embedding(
                input=_input,
                is_sparse=True,
                is_distributed=False,
                size=[
                    self.sparse_feature_number, self.sparse_feature_dim
                ],
                param_attr=paddle.fluid.ParamAttr(
                    name="SparseFeatFactors",
                    initializer=paddle.fluid.initializer.Uniform()
                )
            )

            emb_sum = paddle.fluid.layers.sequence_pool(
                input=emb,
                pool_type='sum'
            )

            return emb_sum

        sparse_embs = list(map(embedding_layer, self.sparse_inputs))

        multi_task_model = self.model(self.config)
        pred_t1, pred_t2, pred_t3 = multi_task_model.forward(sparse_embs)

        pred_t1_1 = paddle.slice(pred_t1, axes=[1], starts=[1], ends=[2])
        pred_t2_1 = paddle.slice(pred_t2, axes=[1], starts=[1], ends=[2])
        pred_t3_1 = paddle.slice(pred_t3, axes=[1], starts=[1], ends=[2])

        auc1, batch_auc_1, auc_state_1 = paddle.static.auc(
            input=pred_t1,
            label=self.t1,
            slide_steps=0
        )

        auc2, batch_auc_2, auc_state_2 = paddle.static.auc(
            input=pred_t2,
            label=self.t2,
            slide_steps=0
        )

        auc3, batch_auc_3, auc_state_3 = paddle.static.auc(
            input=pred_t3,
            label=self.t3,
            slide_steps=0
        )

        if is_infer:
            fetch_dict = {'log_key': self.log_key, 'pred_t1': pred_t1_1, 'pred_t2': pred_t2_1, 'pred_t3': pred_t3_1}
            return fetch_dict

        cost1 = paddle.nn.functional.log_loss(
            input=pred_t1_1,
            label=paddle.cast(
                self.t1,
                dtype="float32"
            )
        )

        cost2 = paddle.nn.functional.log_loss(
            input=pred_t2_1,
            label=paddle.cast(
                self.t2,
                dtype="float32"
            )
        )

        cost3 = paddle.nn.functional.log_loss(
            input=pred_t3_1,
            label=paddle.cast(
                self.t3,
                dtype="float32"
            )
        )

        avg_cost1 = paddle.mean(x=cost1)
        avg_cost2 = paddle.mean(x=cost2)
        avg_cost3 = paddle.mean(x=cost3)

        cost = avg_cost1 + avg_cost2 + avg_cost3

        self._cost = cost
        fetch_dict = {'cost': cost, 'auc_t1': auc1, 'auc_t2': auc2, 'auc_t3': auc3}

        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate,
            lazy_mode=True
        )

        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)

        optimizer.minimize(self._cost)

    def infer_net(self, _input):
        return self.net(_input, is_infer=True)
