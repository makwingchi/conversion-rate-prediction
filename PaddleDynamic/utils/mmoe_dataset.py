import logging

import numpy as np

from paddle.io import IterableDataset

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MMoEDataset(IterableDataset):
    def __init__(self, file_list, config):
        super().__init__()
        self.file_list = file_list
        self.max_len = config["runner"]["max_len"]
        self.init()

    def init(self):
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

        logger.info("pipe init success")

    def __convert(self, conv):
        if conv == "-" or conv == "0":
            return 0
        else:
            return 1

    def __iter__(self):
        for file in self.file_list:
            with open(file, "r") as rf:
                for line in rf:
                    items = line.strip("\n").split("\t")
                    log_key = int(items[0])
                    conv1 = self.__convert(items[1])
                    conv2 = self.__convert(items[2])
                    conv3 = self.__convert(items[3])

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
                                    self.dense_slots_shape[self.slot2index[i]]
                                )
                            else:
                                output[self.slot2index[i]][1].extend([self.padding])
                        else:
                            self.visit[slot] = False

                    res = []
                    for key, value in output:
                        if key == "log_key":
                            continue

                        if key in ["t1", "t2", "t3"]:
                            res.append(np.array(value).astype("float32").reshape([-1,]))
                            continue

                        padding = [0] * (self.max_len - len(value))
                        res.append(
                            np.array(value + padding).astype("int64").reshape([self.max_len,])
                        )

                    len_array = [len(value) for key, value in output][4:]
                    mask = np.array(
                        [[0] * x + [-1e9] * (self.max_len - x) for x in len_array]
                    ).reshape([-1, self.max_len])

                    res.append(mask)
                    yield res
