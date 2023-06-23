import os
import logging
import datetime
import time

import paddle

from models.single_task_model import StaticSingleTaskModel
from utils.utils_single import get_infer_reader, reset_auc
from utils.save_and_load import load_static_model
from utils.train_and_test import dataset_test
from config import get_configurations

LOG_FILE_NAME = f'./logs/infer_log_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt='[%Y-%m-%d %H:%M:%S]',
    filename=LOG_FILE_NAME
)

if __name__ == "__main__":
    config = get_configurations()

    seed = config["runner"]["seed"]
    device = config["runner"]["device"]
    start_epoch = config["runner"]["infer_start_epoch"]
    end_epoch = config["runner"]["infer_end_epoch"]
    model_load_path = config["runner"]["model_save_path"]

    paddle.seed(seed)
    paddle.enable_static()

    # train
    static_model_class = StaticSingleTaskModel(config)
    input_data = static_model_class.create_feeds()
    input_data_names = [data.name for data in input_data]

    place = paddle.set_device(device)

    # infer
    fetch_vars = static_model_class.infer_net(input_data)

    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    dataset, file_list = get_infer_reader(input_data, config)

    for epoch_id in range(start_epoch, end_epoch):
        print("load model epoch {}".format(epoch_id))

        model_path = os.path.join(model_load_path, str(epoch_id))
        load_static_model(
            paddle.static.default_main_program(),
            model_path,
            prefix='rec_static'
        )

        epoch_begin = time.time()
        interval_begin = time.time()
        infer_reader_cost = 0.0
        infer_run_cost = 0.0
        reader_start = time.time()

        reset_auc(False, 1)

        dataset_test(epoch_id, dataset, fetch_vars, exe)
        print("epoch: {} done, ".format(epoch_id) + "epoch time: {:.2f} s".format(time.time() - epoch_begin))
