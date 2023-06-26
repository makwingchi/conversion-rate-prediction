import paddle

from .rec_model import RecModel


class DynamicSingleTaskModel(RecModel):
    def __init__(self, config):
        super().__init__(config)

    def create_feeds(self, batch_data):
        label = batch_data[0]
        features = batch_data[1:-1]
        mask = batch_data[-1]

        return label, features, mask

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
