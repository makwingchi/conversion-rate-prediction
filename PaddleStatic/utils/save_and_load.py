import os

import paddle

from .utils_single import _mkdir_if_not_exist


def load_static_model(program, model_path, prefix='rec_static'):
    print("start load model from {}".format(model_path))
    model_prefix = os.path.join(model_path, prefix)
    param_state_dict = paddle.static.load(program, model_prefix)


def save_static_model(program, model_path, epoch_id, prefix='rec_static'):
    """
    save model to the target path
    """
    model_path = os.path.join(model_path, str(epoch_id))
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    paddle.static.save(program, model_prefix)
    print("Already save model in {}".format(model_path))