import os

import pandas as pd
from sklearn.metrics import roc_auc_score

test_folder = "../work_data/data204194/test_last2"

y_true = []

for _file in os.listdir(test_folder):
    with open(os.path.join(test_folder, _file), "r") as f:
        for line in f:
            items = line.strip().split("\t")

            log_key = items[0]
            conv1 = items[1]
            conv2 = items[2]
            conv3 = items[3]

            if conv1 == "1" or conv2 == "1" or conv3 == "1":
                conv = 1
            else:
                conv = 0

            y_true.append(conv)

pred = pd.read_csv("./test-1.txt", names=["log_key", "pred"])
y_pred = pred["pred"].to_list()

print(roc_auc_score(y_true, y_pred))
