import paddle

from .rec_model import RecModel


class DynamicMultiTaskModel(RecModel):
    def __init__(self, config):
        super().__init__(config)

    def create_feeds(self, batch_data):
        t1 = batch_data[0]
        t2 = batch_data[1]
        t3 = batch_data[2]
        features = batch_data[3:-1]
        mask = batch_data[-1]

        return t1, t2, t3, features, mask

    def create_metrics(self):
        metric_list_name = ["auc_t1", "auc_t2", "auc_t3"]
        metric_list = []

        for i in range(len(metric_list_name)):
            metric_list.append(paddle.metric.Auc("ROC"))

        return metric_list, metric_list_name

    def train_forward(self, model, metric_list, batch_data):
        t1, t2, t3, features, mask = self.create_feeds(batch_data)
        pred1, pred2, pred3 = model.forward(features, mask)

        pred1_1d = paddle.slice(pred1, axes=[1], starts=[1], ends=[2])
        pred2_1d = paddle.slice(pred2, axes=[1], starts=[1], ends=[2])
        pred3_1d = paddle.slice(pred3, axes=[1], starts=[1], ends=[2])

        loss1 = self.create_loss(pred1_1d, t1)
        loss2 = self.create_loss(pred2_1d, t2)
        loss3 = self.create_loss(pred3_1d, t3)

        loss = loss1 + loss2 + loss3

        metric_list[0].update(preds=pred1.numpy(), labels=t1.numpy())
        metric_list[1].update(preds=pred2.numpy(), labels=t2.numpy())
        metric_list[2].update(preds=pred3.numpy(), labels=t3.numpy())

        print_dict = {'loss': loss}

        return loss, metric_list, print_dict

    def infer_forward(self, model, metric_list, batch_data):
        t1, t2, t3, features, mask = self.create_feeds(batch_data)
        pred1, pred2, pred3 = model.forward(features, mask)

        pred1_1d = paddle.slice(pred1, axes=[1], starts=[1], ends=[2])
        pred2_1d = paddle.slice(pred2, axes=[1], starts=[1], ends=[2])
        pred3_1d = paddle.slice(pred3, axes=[1], starts=[1], ends=[2])

        metric_list[0].update(preds=pred1.numpy(), labels=t1.numpy())
        metric_list[1].update(preds=pred2.numpy(), labels=t2.numpy())
        metric_list[2].update(preds=pred3.numpy(), labels=t3.numpy())

        return metric_list, None, pred1_1d.reshape(shape=[-1,]).tolist(), pred2_1d.reshape(shape=[-1,]).tolist(), pred3_1d.reshape(shape=[-1,]).tolist()
