import time
import logging
import argparse

import paddle

from config import get_configurations
from utils.utils_single import create_data_loader
from utils.save_and_load import save_model
from models.single_task_model import DynamicSingleTaskModel
from models.multi_task_model import DynamicMultiTaskModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt='[%Y-%m-%d %H:%M:%S]',
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--conv_type", help="specify conversion type")
parser.add_argument("--purpose", help="specify purpose")
parser.add_argument("--model_type", help="specify model type")
parser.add_argument("--is_infer", help="specify mode", default="0")


if __name__ == "__main__":
    args = parser.parse_args()

    conv_type = str(args.conv_type)
    purpose = str(args.purpose)
    model_type = str(args.model_type)
    is_infer = bool(int(args.is_infer))

    config = get_configurations(conv_type, purpose, model_type, is_infer)
    print(config)

    seed = config["runner"]["seed"]
    device = config["runner"]["device"]
    num_epochs = config["runner"]["train_epochs"]
    print_interval = config["runner"]["print_interval"]
    model_save_path = config["runner"]["model_save_path"]
    task_type = config["runner"]["task_type"]
    is_shuffle = config["runner"]["is_shuffle"]

    paddle.seed(seed)

    if task_type == "single":
        model_class = DynamicSingleTaskModel(config)
    else:
        model_class = DynamicMultiTaskModel(config)

    model = model_class.create_model()
    optimizer = model_class.create_optimizer(model)

    train_dataloader = create_data_loader(config, device, task_type, is_shuffle)

    for epoch_id in range(num_epochs):
        model.train()
        metric_list, metric_list_name = model_class.create_metrics()

        epoch_begin = time.time()
        interval_begin = time.time()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()

        step_num = 0

        for batch_id, batch in enumerate(train_dataloader()):
            train_reader_cost += time.time() - reader_start
            optimizer.clear_grad()
            train_start = time.time()
            batch_size = len(batch[0])

            loss, metric_list, tensor_print_dict = model_class.train_forward(model, metric_list, batch)

            loss.backward()
            optimizer.step()

            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (metric_list_name[metric_id] + ":{:.6f}, ".format(metric_list[metric_id].accumulate()))

                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += ("{}:".format(var_name) + str(var.numpy()).strip("[]") + ",")

                logger.info(
                    "epoch: {}, batch_id: {}, ".format(epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} ins/s".
                    format(
                        train_reader_cost / print_interval,
                        (train_reader_cost + train_run_cost) / print_interval,
                        total_samples / print_interval,
                        total_samples / (train_reader_cost + train_run_cost + 0.0001)
                    )
                )

                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0

            reader_start = time.time()
            step_num += 1

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (metric_list_name[metric_id] + ": {:.6f},".format(metric_list[metric_id].accumulate()))
            metric_list[metric_id].reset()

        tensor_print_str = ""
        if tensor_print_dict is not None:
            for var_name, var in tensor_print_dict.items():
                tensor_print_str += ("{}:".format(var_name) + str(var.numpy()).strip("[]") + ",")

        logger.info(
            "epoch: {} done, ".format(epoch_id) + metric_str + tensor_print_str + " epoch time: {:.2f} s".format(time.time() - epoch_begin)
        )

        save_model(model, optimizer, model_save_path, epoch_id, prefix='rec')
