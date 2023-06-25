import paddle

from .dnn import DNN


class DynamicSingleTaskModel:
    def __init__(self, config):
        self.config = config
        self.model_type = config["runner"]["model_type"]

    def __get_model(self):
        _map = {
            "baseline": DNN
        }

        return _map[self.model_type]

    def create_model(self):
        model = self.__get_model()
        return model(self.config)

    def create_feeds(self, batch_data):
        label = batch_data[0]
        features = batch_data[1:-1]
        mask = batch_data[-1]

        return label, features, mask

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

    def train_forward(self, model, metric_list, batch_data):
        label, features, mask = self.create_feeds(batch_data)
        pred = model.forward(features, mask)

        loss = self.create_loss(pred, label)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metric_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        print_dict = {'loss': loss}

        return loss, metric_list, print_dict

    def infer_forward(self, model, metric_list, batch_data):
        label, features, mask = self.create_feeds(batch_data)
        pred = model.forward(features, mask)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metric_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        return metric_list, None, pred.reshape(shape=[-1,]).tolist()
