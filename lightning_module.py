import torch
import pytorch_lightning as pl
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
# from utils.custom_lr_scheduler import CosineAnnealingWarmupRestarts
from torchmetrics.detection import MeanAveragePrecision

from models import cnn_model, diger
from yattag import Doc      # For sending data to html
from losses.localization_losses import RegionLoss_DIGER
from losses.infer_metrics import PerFrameMetrics
from utils.pred_bbox import PredBbox
from utils.utils import cxcywh_to_xyxy, project_bbox_to_img_coord
from torchviz import make_dot

from typing import Any, List, Tuple, Dict


class LitSupervisedAct(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.l_cfg = cfg.localization_cfg
        self.tg_cfg = cfg.arch_cfg.tg_cfg
        # self.model = cnn_model.CustomResNet(cfg)
        self.model = diger.DIGER(cfg)

        self.save_hyperparameters()
        self.train_loss = RegionLoss_DIGER(cfg)
        self.val_loss = PerFrameMetrics(cfg)
        self.test_loss = PerFrameMetrics(cfg, is_test=True)

        self.test_mAP = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
        self.bbox_pred = PredBbox(cfg)

        self.long_pred_mode = False

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits = self.model(data)

        if self.tg_cfg.type == 'aux_distill':
            if isinstance(logits[-1], tuple):       # For the case of 2D and 3D TG data
                loss = self.tg_cfg.tg_loss_scale * self.train_loss.forward(logits[-1][0], label)
                loss = loss + self.tg_cfg.tg_loss_scale * self.train_loss.forward(logits[-1][1], label)
                loss = loss + self.train_loss.forward((logits[0], logits[1]), label, self.current_epoch)
            else:
                loss = self.tg_cfg.tg_loss_scale * self.train_loss.forward(logits[-1], label[-1])
                loss = loss + self.train_loss.forward((logits[0], logits[1]), label[0], self.current_epoch)
                batch_size = label[0][0].shape[0]
        else:
            loss = self.train_loss.forward(logits, label)
            batch_size = label[0].shape[0]
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        
        logits = self.model(data)
        loss, _, _ = self.val_loss.compute_metrics(logits, label)

        self.log_dict({"val_precision": loss[0], 
                       "val_recall": loss[1], 
                       "val_fscore": loss[2], 
                       "val_acc": loss[3], 
                       "val_loc_recall": loss[4]},
                      prog_bar=True, sync_dist=True, on_epoch=True, batch_size=len(label))

    def test_step(self, batch, batch_idx):
        data, label, frame_names = batch
        batch_losses = []
        conf_mat_pred = []
        conf_mat_label = []
        
        vid_first_fr = [fr[0].name for fr in frame_names]
        
        for idx in range(len(label)):
            logits = self.model(data[idx])
            loss, pred_bboxes, conf_mat_data = self.test_loss.compute_metrics(logits, label[idx])

            # Logging for test epoch end stats
            batch_losses.append(loss)
            conf_mat_pred.extend(conf_mat_data[0])
            conf_mat_label.extend(conf_mat_data[1])

            # Updating the mAP metrics
            pred, target = self.prep_mAP_data(pred_bboxes, label[idx])
            self.test_mAP.update(pred, target)

        ls_sum = lambda input_list, idx: sum([data[idx] for data in input_list]) / len(input_list)

        self.log_dict({"test_precision": ls_sum(batch_losses, 0), 
                       "test_recall": ls_sum(batch_losses, 1), 
                       "test_fscore": ls_sum(batch_losses, 2),
                       "test_acc": ls_sum(batch_losses, 3), 
                       "test_loc_recall": ls_sum(batch_losses, 4)}, 
                       prog_bar=True, sync_dist=True,
                       on_epoch=True, batch_size=len(label))
        
        return batch_losses, vid_first_fr, (conf_mat_pred, conf_mat_label)

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        if self.long_pred_mode:
            data, fr_names = batch
            logits = self.model(data)
            _, pred_bboxes, _ = self.test_loss.compute_metrics(logits, [], predict=True)

            # self.viz_data.save_long_vid_output(pred_bboxes, fr_names)
        else:
            data, label, frame_names = batch

            for idx, data_ in enumerate(data):
                logits = self.model(data_)
                _, pred_bboxes, _ = self.test_loss.compute_metrics(logits, label, predict=True)

                # self.viz_data.gen_vid(pred_bboxes, frame_names[idx])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.train_cfg.max_lr, momentum=0.9)
        warmup_steps = self.num_training_steps * self.cfg.train_cfg.warm_up_epochs
        total_steps = self.num_training_steps * self.cfg.train_cfg.epochs
        lr_sch = CosineAnnealingWarmupRestarts(optimizer,
                                               first_cycle_steps=total_steps,
                                               max_lr=self.cfg.train_cfg.max_lr,
                                               min_lr=self.cfg.train_cfg.min_lr,
                                               warmup_steps=warmup_steps, gamma=self.cfg.train_cfg.gamma)

        scheduler = {
            "scheduler": lr_sch,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def test_epoch_end(self, outputs) -> None:
        
        losses = []
        frames = []
        conf_mat_pred = []
        conf_mat_label = []

        for out in outputs:
            losses.extend(out[0])
            frames.extend(out[1])
            conf_mat_pred.extend(out[2][0])
            conf_mat_label.extend(out[2][1])

        self.logger.experiment.log_html(self.result_html(losses, frames, 'Test Results'))
        self.log_dict(self.test_mAP.compute())
        self.logger.experiment.log_confusion_matrix(y_true=conf_mat_label, 
                                                    y_predicted=conf_mat_pred, 
                                                    title='Test Confusion Matrix',
                                                    labels=self.cfg.OBJECT_ACTION_NAMES.excavator)

    def on_predict_end(self) -> None:
        return super().on_predict_end()

    @staticmethod
    def result_html(metrics, labels, header):
        """
        A script for creating a table for the validation and test results
        to make comparison easier.
        :param clip_frames: The name of the first frame of the input video
        :param labels: The true labels
        :param preds: The prediction results
        :param header: The title of the created table
        :return:
        """
        doc, tag, text = Doc().tagtext()

        with tag('html'):
            with tag('head'):
                with tag('style'):
                    text('table, th, td {border: 1px solid black; border-collapse: collapse;}')

            with tag('body'):
                with tag('h2'):
                    text(header)
                with tag('table'):
                    with tag('tr'):
                        with tag('th'):
                            text('#')
                        with tag('th'):
                            text('Test Video')
                        with tag('th'):
                            text('Precision')
                        with tag('th'):
                            text('Recall')
                        with tag('th'):
                            text('Fscore')
                        with tag('th'):
                            text('Accuracy')
                        with tag('th'):
                            text('Localization Recall')

                    for idx, res in enumerate(zip(labels, metrics)):
                        with tag('tr'):
                            with tag('td'):
                                text(str(idx))
                            with tag('td'):
                                text(str(res[0]))
                            with tag('td'):
                                text(f'{res[1][0]:.3f}')
                            with tag('td'):
                                text(f'{res[1][1]:.3f}')
                            with tag('td'):
                                text(f'{res[1][2]:.3f}')
                            with tag('td'):
                                text(f'{res[1][3]:.3f}')
                            with tag('td'):
                                text(f'{res[1][4]:.3f}')

        return doc.getvalue()

    def load_pretrained_model(self, checkpoint_dir: str) -> None:
        if checkpoint_dir == self.cfg['checkpoint_dir']:
            print("\nNo pre-trained models specified! The model will be randomly initialized.\n")
            return
        
        self.model.load_pretrained_model(checkpoint_dir)
    
    def prep_mAP_data(self, pred: List[torch.Tensor], labels: List[torch.Tensor]) -> Tuple[List[Dict], List[Dict]]:
        """This method reformats the prediction and label data according to the torchmetrics' mAP requirements. 
        For each image (frame), a dict is created which has boxes, scores, and label keys.

        Args:
            pred (_type_): _description_
            labels (_type_): _description_
        """
        out_pred = []
        out_labels = []
        
        for frame, label in zip(pred, labels):
            
            label = label.squeeze(dim=1).transpose(0, 1)    # orig lable shape (5, 1, N_obj)
            label = project_bbox_to_img_coord(cxcywh_to_xyxy(label, self.cfg), self.cfg)
            
            out_pred.append({'boxes': frame[:, :4], 'scores': frame[:, 4], 'labels': frame[:, 5].int()})            
            out_labels.append({'boxes': label[:, :4], 'labels': label[:, -1].int()})
        
        return out_pred, out_labels

    def viz_model_graph(self) -> None:
        batch_size = self.cfg.train_cfg.train_batch_size
        img_size = self.cfg.aug_cfg.resize_size

        if 'distill' in self.tg_cfg.type:
            rgb = torch.randn(batch_size, 3, 16, img_size, img_size, requires_grad=True, device=self.device)
            tg = torch.randn(batch_size, 3, 16, img_size, img_size, requires_grad=True, device=self.device)

            x = (rgb, tg)
        else:
            x = torch.randn(batch_size, 3, 16, img_size, img_size, requires_grad=True, device=self.device)
            
        dot = make_dot(self.model(x), params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True)
        dot.format('png')
        
        img_dir = f"{self.cfg.checkpoint_dir}{self.cfg.exp_datetime}/"
        pth = dot.render(filename='model_graph', directory=img_dir, quiet=True)

        self.logger.experiment.log_image(img_dir + 'model_graph.png')

    def set_long_vid_pred(self, long_vid: bool=False) -> None:
        self.long_pred_mode = long_vid

    @property
    def num_training_steps(self) -> int:
        """
            Total training steps inferred from datamodule and devices.
            Source: https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-774265729
        """
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        dataset = self.trainer._data_connector._train_dataloader_source.dataloader()
        batches = len(dataset)
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices

        # This line gives the total number, what I want is the per epoch one
        # return (batches // effective_accum) * self.trainer.max_epochs
        return batches // effective_accum
