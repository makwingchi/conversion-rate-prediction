import os
import logging

import paddle


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model(model, optimizer, model_path, epoch_id, prefix='rec'):
    model_path = os.path.join(model_path, str(epoch_id))
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    paddle.save(model.state_dict(), model_prefix + ".pdparams")
    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    logger.info("Already save model in {}".format(model_path))


def load_model(model_path, model, prefix="rec"):
    logger.info("start load model from {}".format(model_path))
    model_prefix = os.path.join(model_path, prefix)
    param_state_dict = paddle.load(model_prefix + ".pdparams")
    model.set_dict(param_state_dict)


def _mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)