import os
import datetime
import joblib
import logging
import argparse

from tqdm import tqdm

import paddle
from paddle.io import Dataset, DataLoader

from sklearn.metrics import roc_auc_score


class Reader:
    def __init__(self):
        padding = 0
        sparse_slots = "log_key t1 t2 t3 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = ["dense_feature"]
        self.dense_slots_shape = [13]
        self.slots = self.sparse_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding

    def line_process(self, line):
        items = line.strip("\n").split("\t")
        log_key = int(items[0])
        conv1 = items[1]
        conv2 = items[2]
        conv3 = items[3]

        output = [(i, []) for i in self.slots]
        feasigns = items[4].split(" ")
        for i in feasigns:
            slot_feasign = i.split(":")
            slot = slot_feasign[1]
            if slot not in self.slots:
                continue
            if slot in self.sparse_slots:
                feasign = int(slot_feasign[0])
            else:
                feasign = float(slot_feasign[0])
            output[self.slot2index[slot]][1].append(feasign)
            self.visit[slot] = True
        output[0][1].append(log_key)
        self.visit['log_key'] = True
        output[1][1].append(conv1)
        self.visit['t1'] = True
        output[2][1].append(conv2)
        self.visit['t2'] = True
        output[3][1].append(conv3)
        self.visit['t3'] = True
        for i in self.visit:
            slot = i
            if not self.visit[slot]:
                if i in self.dense_slots:
                    output[self.slot2index[i]][1].extend(
                        [self.padding] *
                        self.dense_slots_shape[self.slot2index[i]])
                else:
                    output[self.slot2index[i]][1].extend([self.padding])
            else:
                self.visit[slot] = False

        return output


class EarlyStopper:
    def __init__(self, num_trials, delta):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = -1
        self.delta = delta
        # self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy + self.delta:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            # torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class CVRDataset(Dataset):
    def __init__(self, features):
        super().__init__()

        self.t1 = []
        self.t2 = []
        self.t3 = []

        self.features = []

        for feature in features:
            if feature[1][0] == "-":
                self.t1.append([0])
            else:
                self.t1.append([int(feature[1][0])])

            if feature[2][0] == "-":
                self.t2.append([0])
            else:
                self.t2.append([int(feature[2][0])])

            if feature[3][0] == "-":
                self.t3.append([0])
            else:
                self.t3.append([int(feature[3][0])])

            self.features.append(feature[4:])

        self.t1 = paddle.to_tensor(self.t1, dtype="float32")
        self.t2 = paddle.to_tensor(self.t2, dtype="float32")
        self.t3 = paddle.to_tensor(self.t3, dtype="float32")

    def __len__(self):
        return len(self.t1)

    def __getitem__(self, idx):
        return self.t1[[idx]], self.t2[[idx]], self.t3[[idx]], self.features[idx]


def train(model, optimizer, data_loader, criterion, log_interval=100):
    model.train()
    total_loss = 0

    tk0 = tqdm(data_loader, smoothing=0, mininterval=1.0)

    for i, (t1, t2, t3, fields) in enumerate(tk0):
        # t1, t2, t3 = t1.to(device), t2.to(device), t3.to(device)
        y = model(fields)

        cat = paddle.concat([t1, t2, t3], axis=1)
        target = paddle.max(cat, axis=1, keepdim=True)

        loss = criterion(y, target)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader):
    model.eval()
    targets, predicts = list(), list()

    with paddle.no_grad():
        for i, (t1, t2, t3, fields) in tqdm(data_loader, smoothing=0, mininterval=1.0):
            # t1, t2, t3 = t1.to(device), t2.to(device), t3.to(device)

            y = model(fields)
            cat = paddle.concat([t1, t2, t3], axis=1)
            target = paddle.max(cat, axis=1, keepdim=True)

            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    return roc_auc_score(targets, predicts)


def collate_fn(data):
    t1, t2, t3, fields = [], [], [], []

    for d in data:
        t1.append(d[0])
        t2.append(d[1])
        t3.append(d[2])

        fields.append(d[3])

    return paddle.concat(t1, axis=0), paddle.concat(t2, axis=0), paddle.concat(t3, axis=0), fields


def run(dataset, model, epoch, learning_rate, batch_size, weight_decay, num_trials, delta, collate_fn=None):
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length

    print(f"training set size: {train_length}, validation set size: {valid_length}, test set size: {test_length}")

    train_dataset, valid_dataset, test_dataset = paddle.io.random_split(dataset, (train_length, valid_length, test_length))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    criterion = paddle.nn.BCELoss()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=num_trials, delta=delta)

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion)

        auc = test(model, valid_data_loader)
        print(f"epoch: {epoch_i}, validation auc: {auc}")

        if not early_stopper.is_continuable(model, auc):
            print(f"validation best auc: {early_stopper.best_accuracy}")
            break

    auc = test(model, test_data_loader)
    print(f"test auc: {auc}")


class DeepCrossingResidualBlock(paddle.nn.Layer):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.linear1 = paddle.nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear2 = paddle.nn.Linear(in_features=hidden_dim, out_features=input_dim)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.linear1(x))
        out2 = self.linear2(out1) + x

        return self.relu(out2)


class DeepCrossing(paddle.nn.Layer):
    def __init__(self, sparse_feature_cnt, sparse_id_num, embedding_dim, hidden_layers):
        super().__init__()

        self.embedding = paddle.nn.Embedding(
            num_embeddings=sparse_id_num,
            embedding_dim=embedding_dim,
        )

        self.residual_layers = paddle.nn.LayerList(
            [DeepCrossingResidualBlock(sparse_feature_cnt * embedding_dim, layer) for layer in hidden_layers]
        )

        self.lin = paddle.nn.Linear(sparse_feature_cnt * embedding_dim, 1)
        self.relu = paddle.nn.ReLU()
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        flattened_list = list(map(self.__flatten_features, x))
        features = paddle.concat(flattened_list)

        for residual_layer in self.residual_layers:
            features = residual_layer(features)

        output = self.lin(features)
        return self.sigmoid(output)

    def __sum_pooling(self, input_list):
        return paddle.sum(input_list, axis=0, keepdim=True)

    def __flatten_features(self, input_list):
        embedded_list = list(map(self.embedding, input_list))
        summed_list = list(map(self.__sum_pooling, embedded_list))

        concatted_list = paddle.concat(summed_list, axis=1)

        return concatted_list


if __name__ == "__main__":
    LOG_FILE_NAME = f'./log_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename=LOG_FILE_NAME
    )

    reader = Reader()

    train_data_path = "./data/data205411/2023-cvr-contest-data/train_data"
    train_data = os.listdir(train_data_path)

    features = []

    for _file in train_data:
        logging.info(f"processing {_file}")
        cnt = 0

        with open(os.path.join(train_data_path, _file)) as f:
            for line in f:
                if cnt % 5000 == 0:
                    logging.info(cnt)

                curr_line = []

                processed_line = reader.line_process(line)

                for _k, _v in processed_line:
                    if _k in ["log_key", "t1", "t2", "t3"]:
                        curr_line.append(_v)
                    else:
                        curr_line.append(paddle.to_tensor(_v, dtype="int64"))

                features.append(curr_line)

                cnt += 1

        new_file_name = _file.replace(".txt", "")
        # joblib.dump(features, os.path.join(train_data_path, new_file_name))
        logging.info(f"{new_file_name} finished")

    paddle.device.set_device("cpu")

    cvr_dataset = CVRDataset(features)

    deepcrossing = DeepCrossing(
        sparse_feature_cnt=26,
        sparse_id_num=88000000,
        embedding_dim=5,
        hidden_layers=[64, 32, 16],
    )

    epoch = 1
    learning_rate = 0.005
    batch_size = 16
    weight_decay = 1e-6
    num_trials = 3
    delta = 0.001

    run(
        dataset=cvr_dataset,
        model=deepcrossing,
        epoch=epoch,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        num_trials=num_trials,
        delta=delta,
        collate_fn=collate_fn
    )
