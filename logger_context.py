import rlpyt.utils.logging.logger as logger
import os.path as osp
import json
from torch.utils.tensorboard.writer import SummaryWriter


def config_logger(log_dir='./bullet_data', name='rlpyt_training', log_params=None, snapshot_mode="last"):
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_log_tabular_only(False)

    run_ID = 0
    while osp.exists(osp.join(log_dir, "run_" + str(run_ID))):
        run_ID += 1
    log_dir = osp.join(log_dir, f"run_{run_ID}")
    exp_dir = osp.abspath(log_dir)
    print('exp_dir: ' + exp_dir)

    tabular_log_file = osp.join(exp_dir, "progress.csv")
    text_log_file = osp.join(exp_dir, "debug.log")
    params_log_file = osp.join(exp_dir, "params.json")

    logger.set_snapshot_dir(exp_dir)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    logger.push_prefix(f"{name}_{run_ID} ")
    logger.set_tf_summary_writer(SummaryWriter(exp_dir))

    if log_params is None:
        log_params = dict()
    log_params["name"] = name
    log_params["run_ID"] = run_ID
    with open(params_log_file, "w") as f:
        json.dump(log_params, f)
