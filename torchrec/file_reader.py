import os
import datetime
import joblib
import logging

import torch

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


if __name__ == "__main__":
    LOG_FILE_NAME = f'./log_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename=LOG_FILE_NAME
    )

    reader = Reader()

    train_data_path = "/mnt/ssd_1/datascience/comp_0713/data/data205411/2023-cvr-contest-data/train_data"
    train_data = os.listdir(train_data_path)

    for _file in train_data:
        logging.info(f"processing {_file}")
        features = []
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
                        curr_line.append(torch.tensor(_v, dtype=torch.int))

                features.append(curr_line)

                cnt += 1

        new_file_name = _file.replace("txt", "pkl")
        joblib.dump(features, os.path.join(train_data_path, new_file_name))
        logging.info(f"{new_file_name} saved")
