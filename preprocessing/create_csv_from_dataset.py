"""
    This script creates a csv file for train and test portions of the dataset
    containing information such as video name and path, object and action class for now.

    Author: Ali Ghelmani,       Date: Dec. 2021
"""
import pandas as pd
from cfg.config import diger_cfg
from pathlib import Path
import numpy as np


def create_train_test_vid_list(config):
    """
        This function looks up all the videos in the dataset and stores their information such as video
        name, video dir, etc. in a list which is then divided into train, eval, and test lists based on
        the ratios provided in the config file. It finally saves the lists into corresponding Excel files
        in the raw video directory.

    :param config: Project config info class containing information such as dataset dir, train ratio, ...
    :return: None
    """
    curr = Path(config.raw_vid_dir)
    total_dataset = []

    for folder in sorted(curr.iterdir()):
        total_dataset.append([folder.name, folder.absolute(), "Excavator", folder.name.split("_")[0].lower()])

    dataset_size = len(total_dataset)
    np.random.seed(0)  # For reproducibility
    rand_perm = np.random.permutation(dataset_size)
    train_idx = int(np.floor(config.train_ratio * dataset_size))
    eval_idx = int(np.floor((config.train_ratio + config.eval_ratio) * dataset_size))

    total_dataset = np.array(total_dataset)
    train_list = total_dataset[rand_perm[:train_idx]]
    eval_list = total_dataset[rand_perm[train_idx:eval_idx]]
    test_list = total_dataset[rand_perm[eval_idx:]]

    csv_headers = config.data_prep.raw_dataset_csv_headers
    save_path = Path(config.ds_excel_dir)
    save_path = save_path.joinpath(f"train-{config.train_ratio}_eval-{config.eval_ratio}_test-{config.test_ratio}")
    save_path.mkdir(exist_ok=True)

    pd.DataFrame(train_list, columns=csv_headers).to_excel(str(save_path) + "/train.xlsx", sheet_name="train_set")
    pd.DataFrame(eval_list, columns=csv_headers).to_excel(str(save_path) + "/eval.xlsx", sheet_name="eval_set")
    pd.DataFrame(test_list, columns=csv_headers).to_excel(str(save_path) + "/test.xlsx", sheet_name="test_set")


if __name__ == "__main__":
    create_train_test_vid_list(diger_cfg)
