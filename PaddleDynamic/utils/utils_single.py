import os

import paddle.io

from .rec_dataset import RecDataset
from .mmoe_dataset import MMoEDataset
from .normal_dataset import CVRDataset


def create_data_loader(config, place, task, is_shuffle, mode="train"):
    if mode == "train":
        data_dir = config["runner"]["train_data_path"]
        batch_size = config["runner"]["batch_size"]
    else:
        data_dir = config["runner"]["test_data_path"]
        batch_size = config["runner"]["batch_size"]

    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

    if task == "single":
        if is_shuffle:
            dataset = CVRDataset(file_list, config)
        else:
            dataset = RecDataset(file_list, config)
    else:
        dataset = MMoEDataset(file_list, config)

    loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        places=place,
        shuffle=is_shuffle,
        drop_last=False
    )

    return loader
