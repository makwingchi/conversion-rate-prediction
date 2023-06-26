import os
import time
import logging

import pandas as pd

import paddle

from utils.utils_single import create_data_loader
from utils.save_and_load import load_model
from config import get_configurations
from models.single_task_model import DynamicSingleTaskModel
from models.multi_task_model import DynamicMultiTaskModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt='[%Y-%m-%d %H:%M:%S]',
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = get_configurations()
    print(config)

    seed = config["runner"]["seed"]
    device = config["runner"]["device"]
    print_interval = config["runner"]["print_interval"]
    start_epoch = config["runner"]["infer_start_epoch"]
    end_epoch = config["runner"]["infer_end_epoch"]
    model_load_path = config["runner"]["model_save_path"]
    infer_batch_size = config["runner"]["batch_size"]
    task_type = config["runner"]["task_type"]

    paddle.seed(seed)

    if task_type == "single":
        model_class = DynamicSingleTaskModel(config)
    else:
        model_class = DynamicMultiTaskModel(config)

    model = model_class.create_model()

    test_dataloader = create_data_loader(config, device, task_type, mode="test")

    epoch_begin = time.time()
    interval_begin = time.time()

    metric_list, metric_list_name = model_class.create_metrics()
    step_num = 0

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, model)

        model.eval()
        infer_reader_cost = 0.0
        infer_run_cost = 0.0
        reader_start = time.time()

        if task_type == "single":
            pred_ls = []
        else:
            pred_ls = {"t1": [], "t2": [], "t3": []}

        for batch_id, batch in enumerate(test_dataloader()):
            infer_reader_cost += time.time() - reader_start
            infer_start = time.time()
            batch_size = len(batch[0])

            if task_type == "single":
                metric_list, tensor_print_dict, curr_pred = model_class.infer_forward(model, metric_list, batch)
                pred_ls.extend(curr_pred)
            else:
                metric_list, tensor_print_dict, pred1, pred2, pred3 = model_class.infer_forward(model, metric_list, batch)
                pred_ls["t1"].extend(pred1)
                pred_ls["t2"].extend(pred2)
                pred_ls["t3"].extend(pred3)

            infer_run_cost += time.time() - infer_start

            if batch_id % print_interval == 0:
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += ("{}:".format(var_name) + str(var.numpy()).strip("[]") + ",")

                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (metric_list_name[metric_id] + ": {:.6f},".format(metric_list[metric_id].accumulate()))

                logger.info(
                    "epoch: {}, batch_id: {}, ".format(epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.2f} ins/s".
                    format(
                        infer_reader_cost / print_interval,
                        (infer_reader_cost + infer_run_cost) / print_interval,
                        infer_batch_size,
                        print_interval * batch_size / (time.time() + 0.0001 - interval_begin)
                    )
                )
                interval_begin = time.time()
                infer_reader_cost = 0.0
                infer_run_cost = 0.0
            step_num = step_num + 1
            reader_start = time.time()

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
        epoch_begin = time.time()

        if task_type == "single":
            pd.DataFrame({"pred": pred_ls}).to_csv("./infer.csv", index=False)
        else:
            pd.DataFrame(pred_ls).to_csv("./infer.csv", index=False)
