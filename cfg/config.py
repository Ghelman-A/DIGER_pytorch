"""
    This class is intended to provide a wrapper for managing the configurations required for running the model
    such as save and load directories, model configurations, hyperparameters, and so on.

    Author: Ali Ghelmani,       Date: June. 15, 2022
"""
import os
import yaml
import datetime
from datetime import datetime as dtime
from pathlib import Path
from munch import munchify

def join_constructor(loader: yaml.SafeLoader, node: yaml.SequenceNode) -> str:
    seq = loader.construct_sequence(node)
    path = ''
    for i in seq:
        path = os.path.join(path, i)
    return path

def tuple_constructor(loader: yaml.SafeLoader, node: yaml.SequenceNode) -> tuple:
    seq = loader.construct_sequence(node)
    return tuple(seq)

def get_loader():
    loader = yaml.SafeLoader
    loader.add_constructor('!join', join_constructor)
    loader.add_constructor('!to_tuple', tuple_constructor)
    return loader

def round_time():
    """A small script for flooring the time at the 2 minute level for avoiding the
    directory name confusion in the lightning ddp.

    Returns:
        str: Rounded datetime at the minute level.
    """
    now = datetime.datetime.now()
    dt = datetime.timedelta(minutes=now.minute % 2, seconds=now.second, microseconds=now.microsecond)
    now -= dt
    return now.strftime("%Y-%m-%d %H:%M:%S")[:-3]

def sync_checkpoint_dir(cfg) -> str:
    """
        This short script is for creating the checkpoint directory due to the problem in lightning
        in which each process in ddp mode will create a different directory based on the starting time,
        which I have to say is very inconvenient!

    Args:
        cfg (Dict): The DIGER config dictionary
    """

    cfg['exp_datetime'] = round_time()
    curr = Path(f"{cfg['checkpoint_dir']}{cfg['exp_datetime']}")

    dt = datetime.timedelta(minutes=2)
    folder_name_dtime = dtime.strptime(cfg['exp_datetime'], "%Y-%m-%d %H:%M")
    next_folder_name = (folder_name_dtime + dt).strftime("%Y-%m-%d %H:%M")
    prev_folder_name = (folder_name_dtime - dt).strftime("%Y-%m-%d %H:%M")
    
    if curr.with_name(next_folder_name).exists():
        cfg['exp_datetime'] = next_folder_name

    elif curr.with_name(prev_folder_name).exists():
        cfg['exp_datetime'] = prev_folder_name
    
    os.makedirs(f"{cfg['checkpoint_dir']}{cfg['exp_datetime']}", exist_ok=True)
    os.makedirs(f"{cfg['infer_output_dir']}{cfg['exp_datetime']}", exist_ok=True)
    return cfg['exp_datetime']

with open('cfg/config.yaml', 'rb') as f:
    diger_cfg = yaml.load(f, Loader=get_loader())

diger_cfg['exp_datetime'] = sync_checkpoint_dir(diger_cfg)
os.makedirs(diger_cfg['checkpoint_dir'], exist_ok=True)
os.makedirs(diger_cfg['ds_list_dir'], exist_ok=True)
os.makedirs(diger_cfg['ds_excel_dir'], exist_ok=True)

diger_cfg = munchify(diger_cfg)

#---------------------------------------------------------------#
#                  Loading the DIGER config                      #
#---------------------------------------------------------------#
with open(diger_cfg.arch_cfg, 'rb') as f:
    diger_cfg.arch_cfg = munchify(yaml.load(f, Loader=get_loader()))
