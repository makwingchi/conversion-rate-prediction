import os
import datetime
import joblib
import logging

from tqdm import tqdm

import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score


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


class CVRDataset(torch.utils.data.Dataset):
    def __init__(self, features):
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

        self.t1 = torch.tensor(self.t1).double()
        self.t2 = torch.tensor(self.t2).double()
        self.t3 = torch.tensor(self.t3).double()

    def __len__(self):
        return len(self.t1)

    def __getitem__(self, idx):
        return self.t1[[idx]], self.t2[[idx]], self.t3[[idx]], self.features[idx]


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0

    tk0 = tqdm(data_loader, smoothing=0, mininterval=1.0)

    for i, (t1, t2, t3, fields) in enumerate(tk0):
        t1, t2, t3 = t1.to(device), t2.to(device), t3.to(device)
        y = model(fields)

        cat = torch.cat([t1, t2, t3], dim=1)
        target = torch.max(cat, dim=1, keepdim=True).values

        loss = criterion(y, target)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()

    with torch.no_grad():
        for i, (t1, t2, t3, fields) in tqdm(data_loader, smoothing=0, mininterval=1.0):
            t1, t2, t3 = t1.to(device), t2.to(device), t3.to(device)

            y = model(fields)
            cat = torch.cat([t1, t2, t3], dim=1)
            target = torch.max(cat, dim=1, keepdim=True).values

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

    return torch.cat(t1, dim=0), torch.cat(t2, dim=0), torch.cat(t3, dim=0), fields


def run(dataset, model, device, epoch, learning_rate, batch_size, weight_decay, num_trials, delta, collate_fn=None):
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length

    logging.info(f"training set size: {train_length}, validation set size: {valid_length}, test set size: {test_length}")

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, valid_length, test_length))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=num_trials, delta=delta)

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)

        auc = test(model, valid_data_loader, device)
        logging.info(f"epoch: {epoch_i}, validation auc: {auc}")

        if not early_stopper.is_continuable(model, auc):
            logging.info(f"validation best auc: {early_stopper.best_accuracy}")
            break

    auc = test(model, test_data_loader, device)
    logging.info(f"test auc: {auc}")


class DeepCrossingResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()

        self.linear1 = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, dtype=torch.double, device=device)
        self.linear2 = torch.nn.Linear(in_features=hidden_dim, out_features=input_dim, dtype=torch.double, device=device)

    def forward(self, x):
        out1 = torch.relu(self.linear1(x))
        out2 = self.linear2(out1) + x

        return torch.relu(out2)


class DeepCrossing(torch.nn.Module):
    def __init__(self, sparse_feature_cnt, sparse_id_num, embedding_dim, hidden_layers, device):
        super().__init__()

        self.device = device

        self.embedding = torch.nn.Embedding(
            num_embeddings=sparse_id_num,
            embedding_dim=embedding_dim,
            dtype=torch.double,
            device=device
        )

        self.residual_layers = torch.nn.ModuleList(
            [DeepCrossingResidualBlock(sparse_feature_cnt * embedding_dim, layer, device) for layer in hidden_layers]
        )

        self.lin = torch.nn.Linear(sparse_feature_cnt * embedding_dim, 1, dtype=torch.double, device=device)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        flattened_list = list(map(self.__flatten_features, x))
        features = torch.cat(flattened_list).to(self.device)

        for residual_layer in self.residual_layers:
            features = residual_layer(features)

        output = self.lin(features)
        return torch.sigmoid(output)

    def __sum_pooling(self, input_list):
        return torch.sum(input_list, dim=0, keepdim=True)

    def __flatten_features(self, input_list):
        embedded_list = list(map(self.embedding, input_list))
        summed_list = list(map(self.__sum_pooling, embedded_list))

        concatted_list = torch.cat(summed_list, dim=1)

        return concatted_list


if __name__ == "__main__":
    LOG_FILE_NAME = f'./training_log_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename=LOG_FILE_NAME
    )

    clean_train_folder = "/mnt/ssd_1/datascience/comp_0713/data/data205411/2023-cvr-contest-data/clean_train"

    train_file_ls = os.listdir(clean_train_folder)
    train_file_ls = [_file for _file in train_file_ls if not _file.startswith("train")]

    train_df_ls = []

    for _file in tqdm(train_file_ls[:1]):
        curr_train = joblib.load(os.path.join(clean_train_folder, _file))
        train_df_ls.append(curr_train)

    col_names = [
        'sample_id', 't1', 't2', 't3', 'user_feat_1', 'user_feat_2',
        'user_feat_3', 'user_feat_4', 'user_feat_5', 'user_feat_6',
        'user_feat_7', 'user_feat_8', 'user_feat_9', 'user_feat_10',
        'user_feat_11', 'user_feat_12', 'user_feat_13', 'scene_feat_14',
        'scene_feat_15', 'scene_feat_16', 'ad_feat_17', 'ad_feat_18',
        'ad_feat_19', 'ad_feat_20', 'ad_feat_21', 'ad_feat_22', 'ad_feat_23',
        'ad_feat_24', 'session_feat_25', 'session_feat_26'
    ]

    for _df in train_df_ls:
        _df.drop("index", axis=1, inplace=True, errors="ignore")
        _df.columns = col_names

    train_df = pd.concat(train_df_ls, axis=0, ignore_index=True)

    features = []
    feature_cols = list(train_df.columns[4:])

    for idx, row in train_df.head(1000).iterrows():
        curr_features = []

        curr_features.append([row["sample_id"]])
        curr_features.append([row["t1"]])
        curr_features.append([row["t2"]])
        curr_features.append([row["t3"]])

        for _col in feature_cols:
            if len(row[_col]) == 0:
                curr_features.append(torch.tensor([0], dtype=torch.long))
            else:
                tmp = []
                for i in row[_col]:
                    tmp.append(int(i))

                curr_features.append(torch.tensor(tmp, dtype=torch.long))

        features.append(curr_features)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 1000
    learning_rate = 0.005
    batch_size = 16
    weight_decay = 1e-6
    num_trials = 3
    delta = 0.001

    cvr_dataset = CVRDataset(features)

    deepcrossing = DeepCrossing(
        sparse_feature_cnt=26,
        sparse_id_num=88000000,
        embedding_dim=5,
        hidden_layers=[32, 16, 8],
        device=device
    )

    run(
        dataset=cvr_dataset,
        model=deepcrossing,
        device=device,
        epoch=epoch,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        num_trials=num_trials,
        delta=delta,
        collate_fn=collate_fn
    )
