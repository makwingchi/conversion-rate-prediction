import os

import pandas as pd

test_folder = "../work_data/data204194/test_data"

test_records = {"log_key": [], "ad_type": []}

for _file in os.listdir(test_folder):
    with open(os.path.join(test_folder, _file), "r") as f:
        for line in f:
            items = line.strip().split("\t")

            log_key = items[0]
            test_records["log_key"].append(log_key)

            features = items[4]
            feature_ls = features.split(" ")

            for _f in feature_ls:
                _key = _f.split(":")[1]

                if _key != "9":
                    continue

                _value = _f.split(":")[0]

                if _value in ['2367', '321']:
                    test_records["ad_type"].append(1)
                elif _value in ['167', '350']:
                    test_records["ad_type"].append(2)
                elif _value in ['75', '9']:
                    test_records["ad_type"].append(3)

test_df = pd.DataFrame(test_records)

pred1 = pd.read_csv("./infer_1.csv")
pred2 = pd.read_csv("./infer_2.csv")
pred3 = pd.read_csv("./infer_3.csv")

test_df.loc[test_df.ad_type == 1, "pred"] = pred1["pred"].values
test_df.loc[test_df.ad_type == 2, "pred"] = pred2["pred"].values
test_df.loc[test_df.ad_type == 3, "pred"] = pred3["pred"].values

test_df[["log_key", "pred"]].to_csv("./test-1.txt", index=False, header=False)
