import os

import pandas as pd

from config import get_configurations

config = get_configurations()

test_data_path = config["runner"]["test_data_path"]
test_data = os.path.join(test_data_path, "file_test.txt")

task_type = config["runner"]["task_type"]

infer = pd.read_csv("infer.csv")

log_keys = []

with open(test_data, 'r') as f:
    for line in f:
        items = line.strip("\n").split("\t")
        log_key = int(items[0])

        log_keys.append(log_key)

infer["log_key"] = log_keys

if task_type == "multi":
    infer["pred"] = infer.apply(lambda x: max(x["t1"], x["t2"], x["t3"]), axis=1)

infer = infer[["log_key", "pred"]]

infer.to_csv("test-1.txt", index=False, header=False)
