import paddle

from .dnn import DNN
from .deepcrossing import DeepCrossing
from .wideanddeep import WideAndDeep
from .deepandcross import DeepAndCross
from .fnn import FNN
from .pnn import PNN
from .dlrm import DLRM
from .mmoe import MMoE
from .naive_attention import NativeAttention


class RecModel:
    def __init__(self, config):
        self.config = config
        self.model_type = config["runner"]["model_type"]

    def __get_model(self):
        _map = {
            "baseline": DNN,
            "wideanddeep": WideAndDeep,
            "deepandcross": DeepAndCross,
            "deepcrossing": DeepCrossing,
            "pnn": PNN,
            "fnn": FNN,
            "mmoe": MMoE,
            "dlrm": DLRM,
            "naive_attention": NativeAttention
        }

        return _map[self.model_type]

    def create_model(self):
        model = self.__get_model()
        return model(self.config)

    def create_loss(self, pred, label):
        cost = paddle.nn.functional.log_loss(
            input=pred,
            label=paddle.cast(label, dtype="float32")
        )

        return paddle.mean(x=cost)

    def create_optimizer(self, model):
        learning_rate = self.config["optimizer"]["learning_rate"]
        optimizer = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            parameters=model.parameters()
        )

        return optimizer

    def create_metrics(self):
        metric_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC")
        metric_list = [auc_metric]

        return metric_list, metric_list_name
