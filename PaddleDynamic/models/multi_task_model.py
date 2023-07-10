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

    def create_loss(self, pred, label):
        loss_function = paddle.nn.CrossEntropyLoss(soft_label=True)
        cost = loss_function(pred, label)

        return cost

    def train_forward(self, model, metric_list, batch_data):
        t1, t2, t3, features, mask = self.create_feeds(batch_data)
        pred1, pred2, pred3 = model.forward(features, mask)

        pred1_2d = paddle.slice(pred1, axes=[1], starts=[1], ends=[3])
        pred2_2d = paddle.slice(pred2, axes=[1], starts=[1], ends=[3])
        pred3_2d = paddle.slice(pred3, axes=[1], starts=[1], ends=[3])

        t1_1d = paddle.slice(t1, axes=[1], starts=[2], ends=[3])
        t2_1d = paddle.slice(t2, axes=[1], starts=[2], ends=[3])
        t3_1d = paddle.slice(t3, axes=[1], starts=[2], ends=[3])

        loss1 = self.create_loss(pred1, t1)
        loss2 = self.create_loss(pred2, t2)
        loss3 = self.create_loss(pred3, t3)

        loss = loss1 + loss2 + loss3

        metric_list[0].update(preds=pred1_2d.numpy(), labels=t1_1d.numpy())
        metric_list[1].update(preds=pred2_2d.numpy(), labels=t2_1d.numpy())
        metric_list[2].update(preds=pred3_2d.numpy(), labels=t3_1d.numpy())

        print_dict = {'loss': loss}

        return loss, metric_list, print_dict

    def infer_forward(self, model, metric_list, batch_data):
        t1, t2, t3, features, mask = self.create_feeds(batch_data)
        pred1, pred2, pred3 = model.forward(features, mask)

        pred1_1d = paddle.slice(pred1, axes=[1], starts=[2], ends=[3])
        pred2_1d = paddle.slice(pred2, axes=[1], starts=[2], ends=[3])
        pred3_1d = paddle.slice(pred3, axes=[1], starts=[2], ends=[3])

        pred1_2d = paddle.slice(pred1, axes=[1], starts=[1], ends=[3])
        pred2_2d = paddle.slice(pred2, axes=[1], starts=[1], ends=[3])
        pred3_2d = paddle.slice(pred3, axes=[1], starts=[1], ends=[3])

        t1_1d = paddle.slice(t1, axes=[1], starts=[2], ends=[3])
        t2_1d = paddle.slice(t2, axes=[1], starts=[2], ends=[3])
        t3_1d = paddle.slice(t3, axes=[1], starts=[2], ends=[3])

        metric_list[0].update(preds=pred1_2d.numpy(), labels=t1_1d.numpy())
        metric_list[1].update(preds=pred2_2d.numpy(), labels=t2_1d.numpy())
        metric_list[2].update(preds=pred3_2d.numpy(), labels=t3_1d.numpy())

        return metric_list, None, pred1_1d.reshape(shape=[-1,]).tolist(), pred2_1d.reshape(shape=[-1,]).tolist(), pred3_1d.reshape(shape=[-1,]).tolist()
