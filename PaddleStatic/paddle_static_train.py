import datetime
import logging

import paddle

from models.single_task_model import StaticSingleTaskModel
from models.multi_task_model import StaticMultiTaskModel
from utils.utils_single import get_reader, reset_auc
from utils.save_and_load import save_static_model
from utils.train_and_test import dataset_train
from config import get_configurations


LOG_FILE_NAME = f'./logs/train_log_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt='[%Y-%m-%d %H:%M:%S]',
    filename=LOG_FILE_NAME
)


if __name__ == "__main__":
    config = get_configurations()
    print(config)

    seed = config["runner"]["seed"]
    model_save_path = config["runner"]["model_save_path"]
    device = config["runner"]["device"]
    num_epochs = config["runner"]["train_epochs"]
    task_type = config["runner"]["task_type"]

    paddle.seed(seed)
    paddle.enable_static()

    # train
    # load static model class
    if task_type == "single":
        static_model_class = StaticSingleTaskModel(config)
    else:
        static_model_class = StaticMultiTaskModel(config)

    input_data = static_model_class.create_feeds()
    input_data_names = [data.name for data in input_data]

    fetch_vars = static_model_class.net(input_data)

    place = paddle.set_device(device)
    static_model_class.create_optimizer()

    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    dataset, file_list = get_reader(input_data, config)

    for epoch_id in range(num_epochs):
        reset_auc(False, 1)

        dataset_train(epoch_id, dataset, fetch_vars, exe)

        save_static_model(
            paddle.static.default_main_program(),
            model_save_path,
            epoch_id,
            prefix='rec_static'
        )
