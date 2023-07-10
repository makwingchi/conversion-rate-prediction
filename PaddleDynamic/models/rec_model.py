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
from .deepcrossingV2 import DeepCrossingV2
from .deepfm import DeepFM
from .nfm import NFM
from .afm import AFM
from .autoint import AutoInt
from .deepcrossingattn import DeepCrossingAttn
from .dcnv3 import DeepAndCrossV3
from .dcnv5 import DeepAndCrossV5
from .dcnv8 import DeepAndCrossV8
from .dcnv9 import DeepAndCrossV9
from .dcnv14 import DeepAndCrossV14
from .share_bottom import ShareBottom


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
            "deepcrossingV2": DeepCrossingV2,
            "deepfm": DeepFM,
            "nfm": NFM,
            "afm": AFM,
            "autoint": AutoInt,
            "naive_attention": NativeAttention,
            "deepcrossingattn": DeepCrossingAttn,
            "deepandcrossv3": DeepAndCrossV3,
            "deepandcrossv5": DeepAndCrossV5,
            "deepandcrossv8": DeepAndCrossV8,
            "deepandcrossv9": DeepAndCrossV9,
            "deepandcrossv14": DeepAndCrossV14,
            "sharebottom": ShareBottom
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
