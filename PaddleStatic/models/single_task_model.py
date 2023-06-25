import paddle

from .dnn import DNN
from .deepcrossing import DeepCrossing
from .wideanddeep import WideAndDeep
from .deepandcross import DeepAndCross
from .fnn import FNN
from .pnn import PNN
from .deepfm import DeepFM
from .nfm import NFM
from .fibinet import FiBiNet


class StaticSingleTaskModel:
    def __init__(self, config):
        self.cost = None
        self.infer_target_var = None
        self.config = config
        self._init_hyper_parameters()
        self._get_model()

    def _init_hyper_parameters(self):
        self.model_type = self.config["runner"]["model_type"].lower()

        self.sparse_inputs_slots = self.config["models"][self.model_type]["sparse_inputs_slots"]
        self.sparse_feature_number = self.config["models"][self.model_type]["sparse_feature_number"]
        self.sparse_feature_dim = self.config["models"][self.model_type]["sparse_feature_dim"]
        self.learning_rate = self.config["optimizer"]["learning_rate"]

    def _get_model(self):
        _map = {
            "baseline": DNN,
            "wideanddeep": WideAndDeep,
            "deepandcross": DeepAndCross,
            "deepcrossing": DeepCrossing,
            "deepfm": DeepFM,
            "pnn": PNN,
            "fnn": FNN,
            "nfm": NFM,
            "fibinet": FiBiNet
        }

        self.model = _map[self.model_type]

    def create_feeds(self):
        sparse_input_ids = [
            paddle.static.data(
                name="C" + str(i), shape=[None, 1], lod_level=1, dtype="int64")
            for i in range(1, self.sparse_inputs_slots)
        ]

        label = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64"
        )

        feeds_list = [label] + sparse_input_ids

        return feeds_list

    def net(self, _input, is_infer=False):
        self.log_key = _input[0]
        self.label_input = _input[1]
        self.sparse_inputs = _input[2:self.sparse_inputs_slots]

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

        dnn_model = self.model(self.config)
        pred = dnn_model.forward(sparse_embs)
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)

        auc, batch_auc_var, _ = paddle.static.auc(
            input=predict_2d,
            label=self.label_input,
            slide_steps=0
        )
        self.inference_target_var = auc

        if is_infer:
            fetch_dict = {'log_key': self.log_key, 'pred': pred}
            return fetch_dict

        cost = paddle.nn.functional.log_loss(
            input=pred,
            label=paddle.cast(
                self.label_input,
                dtype="float32"
            )
        )

        avg_cost = paddle.mean(x=cost)
        self._cost = avg_cost
        fetch_dict = {'cost': avg_cost, 'auc': auc}

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
