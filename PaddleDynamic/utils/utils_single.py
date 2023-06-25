import os

import paddle.io

from .rec_dataset import RecDataset


def create_data_loader(config, place, mode="train"):
    if mode == "train":
        data_dir = config["runner"]["train_data_path"]
        batch_size = config["runner"]["batch_size"]
        shuffle = True
    else:
        data_dir = config["runner"]["test_data_path"]
        batch_size = config["runner"]["batch_size"]
        shuffle = False

    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    dataset = RecDataset(file_list, config)

    loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        places=place,
        shuffle=shuffle,
        drop_last=False
    )

    return loader
