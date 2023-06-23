import os

import numpy as np

import paddle

from .queue import Queue


def get_filepath(dir_path, list_name):
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)

        if os.path.isdir(file_path):
            get_filepath(file_path, list_name)
        else:
            list_name.append(file_path)

    return list_name


def get_file_list(data_path):
    assert os.path.exists(data_path)
    list_name = []
    file_list = get_filepath(data_path, list_name)

    print("File list: {}".format(file_list))

    return file_list


def get_reader(input_var, config):
    # train_data_path = "../data/data205411/2023-cvr-contest-data/train_data"
    # train_data_path = "../../../../data/data205411/2023-cvr-contest-data/train_data"
    train_data_path = config["runner"]["train_data_path"]

    file_list = get_file_list(train_data_path)
    print("train file_list: {}".format(file_list))

    reader_instance = Queue(input_var, file_list, config)
    return reader_instance.get_reader(), file_list


def get_infer_reader(input_var, config):
    # test_data_path = "../data/data204194/test_data"
    # test_data_path = "../../../../data/data204194/test_data"
    test_data_path = config["runner"]["test_data_path"]

    print("test_data_path is: {}".format(test_data_path))
    file_list = get_file_list(test_data_path)
    print("test file_list: {}".format(file_list))

    reader_instance = Queue(input_var, file_list, config)
    return reader_instance.get_infer_reader(), file_list


def _mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def reset_auc(use_fleet=False, auc_num=1):
    # for static clear auc
    auc_var_name = []
    for i in range(auc_num * 5):
        auc_var_name.append("_generated_var_{}".format(i))

    for name in auc_var_name:
        param = paddle.static.global_scope().find_var(name)
        if param == None:
            continue
        tensor = param.get_tensor()
        if param:
            tensor_array = np.zeros(tensor._get_dims()).astype("int64")
            if use_fleet:
                trainer_id = paddle.distributed.get_rank()
                tensor.set(tensor_array, paddle.CUDAPlace(trainer_id))
            else:
                tensor.set(tensor_array, paddle.CPUPlace())
            print("AUC Reset To Zero: {}".format(name))
