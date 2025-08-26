
import comet_ml  # Had to add it due to an error with comet logger
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler, PyTorchProfiler

from cfg.config import diger_cfg

from lightning_module import LitSupervisedAct
from equipmentDataModule import EquipmentDataModule

def main(cfg):
    pl.seed_everything(seed=1234, workers=True)
    tr_mode = cfg.train_mode

    #------ Setting the lightning trainer
    dm = EquipmentDataModule(cfg)
    model = LitSupervisedAct(cfg)

    #------ Setting the Comet logger and profiler
    comet_logger = CometLogger(
        api_key="",
        workspace="",  # Optional
        project_name="",  # Optional
        experiment_name=f'{cfg["exp_datetime"]}'
    )
    comet_logger.log_graph(model)  # Record model graph
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    
    if cfg[tr_mode]['stage'] in ['train']:
        #------ Setting the required callbacks and profiler
        lr_mon = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(f"{cfg.checkpoint_dir}{cfg.exp_datetime}", 
                                              monitor=cfg[tr_mode]['monitor'], 
                                              mode=cfg[tr_mode]['mode'])
        
        early_stop = EarlyStopping(monitor=cfg[tr_mode]['monitor'], 
                                   mode=cfg[tr_mode]['mode'], 
                                   patience=cfg[tr_mode]['patience'],
                                   min_delta=cfg[tr_mode]['min_delta'])

        #------ Training!
        trainer = pl.Trainer(gpus=cfg.train_cfg.gpu_device_ids, 
                             strategy=DDPStrategy(find_unused_parameters=False), 
                             logger=[comet_logger],
                             log_every_n_steps=1, 
                             min_epochs=cfg.train_cfg.epochs, 
                             max_epochs=100, 
                             precision=32, 
                             profiler=None,
                             callbacks=[lr_mon, ModelSummary(max_depth=100), checkpoint_callback, early_stop])
        
        trainer.fit(model, datamodule=dm)

        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:

            trainer = pl.Trainer(gpus=[1], logger=[comet_logger], log_every_n_steps=1, precision=32, profiler=None)
            model = LitSupervisedAct.load_from_checkpoint(checkpoint_callback.best_model_path)

            trainer.test(model, datamodule=dm)

    elif cfg[tr_mode]['stage'] in ['test']:

        trainer = pl.Trainer(gpus=[1], logger=[comet_logger], log_every_n_steps=1, precision=32, profiler=None)
        if trainer.is_global_zero:

            model.load_pretrained_model(cfg[tr_mode]['checkpoint_dir'])
            trainer.test(model, datamodule=dm)
    
    elif cfg[tr_mode]['stage'] in ['predict']:

        pred_long_vid = True

        if not pred_long_vid:
            pred_vid_list = []      # If empty, predict the entire dataset, enter only the name of video
            dm.set_pred_data(pred_vid_list)
        else:
            dm.set_pred_long_data(pred_long_vid)
            model.set_long_vid_pred(pred_long_vid)

        trainer = pl.Trainer(gpus=[1], logger=[comet_logger], precision=16, profiler=None)
        if trainer.is_global_zero:

            model.load_pretrained_model(cfg[tr_mode]['checkpoint_dir'])
            trainer.predict(model, datamodule=dm)


if __name__ == "__main__":
    main(diger_cfg)
