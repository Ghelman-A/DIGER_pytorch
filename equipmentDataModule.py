"""
    This scripts wraps most of the data processing steps into a LightningDataModule for ease
    of reproducibility and use.

    Author: Ali Ghelmani,       Last Modified: June 12, 2022
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import albumentations as A
from torchvideotransforms import video_transforms, volume_transforms

from preprocessing.customdatasets import SupervisedDataset
from preprocessing.dataprocessing import PrepareDataLists
from utils.custom_sampler import PredictSampler
from typing import List


class EquipmentDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.l_cfg = cfg.localization_cfg
        self.long_vid_cfg = cfg.long_vid_prep
        self.tg_mode = bool(cfg.arch_cfg.tg_cfg.type)       # If type == '' no TG data is available
        self.prepare_data()

        self.num_workers = 4

        self.train_transform = self.train_transforms()
        self.test_transform = self.test_transforms()

        # Assigning the path for the train and evaluation files
        self.train_mode = cfg.train_mode
        self.train_pkl  = cfg[self.train_mode]['train_pkl_dir']
        self.val_pkl    = cfg[self.train_mode]["eval_pkl_dir"]
        self.test_pkl   = cfg.test_pkl_dir
        self.pred_pkl   = cfg.predict_pkl_dir

        # Variables that will be initiated in the setup method
        self.train_ds   = None
        self.val_ds     = None
        self.test_ds    = None
        self.pred_ds    = None

        # Variables for the long video prediction mode
        self.long_pred_mode = False

    def prepare_data(self):
        if self.cfg.data_prep.prep_needed:
            data_transform = PrepareDataLists(self.cfg)
            data_transform.train()
            data_transform.predict()

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            self.train_ds   = SupervisedDataset(self.cfg, self.train_pkl, self.train_transform)
            self.val_ds     = SupervisedDataset(self.cfg, self.val_pkl, self.test_transform, mode='val')
                
        elif stage in (None, 'test'):
            self.test_ds = SupervisedDataset(self.cfg, self.test_pkl, self.test_transform, mode='test')

        elif stage in (None, 'predict'):
            if not self.long_pred_mode:
                self.pred_ds = SupervisedDataset(self.cfg, self.pred_pkl, self.test_transform, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.cfg.train_cfg.train_batch_size, shuffle=True, drop_last=False, 
                          num_workers=self.num_workers, collate_fn=self.bbox_collate, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.cfg.train_cfg.val_batch_size, shuffle=False, drop_last=False,
                              num_workers=self.num_workers, collate_fn=self.val_bbox_collate, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.cfg.train_cfg.test_batch_size, shuffle=False, drop_last=False,
                          num_workers=self.num_workers, collate_fn=self.test_bbox_collate, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_ds, self.cfg.train_cfg.test_batch_size, shuffle=False, drop_last=False, 
                          sampler=self.pred_sampler,
                          num_workers=self.num_workers, collate_fn=self.test_bbox_collate, pin_memory=True)

    def train_transforms(self):
        in_size = self.cfg.aug_cfg.resize_size
        br  = self.cfg.aug_cfg.jitter_brightness
        ctr = self.cfg.aug_cfg.jitter_contrast
        sat = self.cfg.aug_cfg.jitter_saturation
        hue = self.cfg.aug_cfg.jitter_hue

        add_targets = {}
        tot_frames = 17 if self.tg_mode else 16
        [add_targets.update({f'image{i+1}': 'image'}) for i in range(tot_frames - 1)]
        [add_targets.update({f'bboxes{i+1}': 'bboxes'}) for i in range(tot_frames - 1)]

        transform_1 = A.Compose([
            # A.RandomSizedBBoxSafeCrop(height=in_size, width=in_size, erosion_rate=0),
            A.Resize(height=in_size, width=in_size),
            A.HorizontalFlip(p=self.cfg.aug_cfg.flip_chance),
            A.ColorJitter(brightness=br, contrast=ctr, saturation=sat, hue=hue),
            A.ToGray(p=self.cfg.aug_cfg.greyscale_chance),
            A.GaussianBlur(blur_limit=self.cfg.aug_cfg.gaussian_blur_kernel),
            A.OneOf([
                A.RandomRotate90(),
                ], p=0.8),
            A.OneOf([
                A.Solarize(),
                ], p=0.8),
            A.GaussNoise(),
            A.OneOf([
                A.MotionBlur(),
                A.GaussianBlur(blur_limit=self.cfg.aug_cfg.gaussian_blur_kernel),
                A.Blur(),
                A.ZoomBlur(),
                ], p=0.8),
            A.Cutout(),
            A.Normalize(mean=self.cfg.aug_cfg.mean, std=self.cfg.aug_cfg.std),
        ], bbox_params = A.BboxParams(format='yolo', label_fields=['category_ids']), 
        additional_targets=add_targets)

        if self.tg_mode:
            transform_2 = A.Compose([
                A.RandomSizedBBoxSafeCrop(height=in_size, width=in_size, erosion_rate=0),
                # A.Resize(height=in_size, width=in_size),
            ],  bbox_params = A.BboxParams(format='yolo', label_fields=['category_ids']), 
        additional_targets=add_targets)
            
            return transform_1, transform_2
        else:
            return transform_1        
    
    def test_transforms(self):

        in_size = self.cfg.aug_cfg.resize_size
        transform_1 = [video_transforms.Resize(size=(in_size, in_size)),
                       volume_transforms.ClipToTensor()]
        transform_2 = [video_transforms.Normalize(mean=self.cfg.aug_cfg.mean, std=self.cfg.aug_cfg.std)]

        if self.cfg.arch_cfg.tg_cfg.type == 'concat':       #! Deprecated!
            return video_transforms.Compose(transform_1), video_transforms.Compose(transform_2)

        return video_transforms.Compose(transform_1 + transform_2)

    def set_pred_data(self, pred_list: List[str]=None):
        self.pred_sampler = PredictSampler(self.pred_ds, pkl_file=self.pred_pkl, pred_list=pred_list)

    def set_pred_long_data(self, long_vid: bool=False):
        self.long_pred_mode = long_vid

    @staticmethod
    def bbox_collate(data_list):
        """
            This method creates the ground truthes required for calculating the localization loss of the bounding boxes.

        Args:
            data_list (_type_): _description_
        """
        if len(data_list[0][0]) == 2:
            rgb_data    = torch.stack([data_[0][0] for data_ in data_list])
            tg_data     = torch.stack([data_[0][1] for data_ in data_list])
            
            tg_xywh_mask   = torch.stack([data_[2][0] for data_ in data_list])
            tg_cls_mask    = torch.stack([data_[2][1] for data_ in data_list])
            
            tg_max_obj = [data_[2][2].shape[-1] for data_ in data_list]
            tg_max_obj = max(tg_max_obj)
            tg_bbox_gt = torch.zeros((rgb_data.shape[0], 4, 1, tg_max_obj), 
                                     device=rgb_data.device).index_fill_(dim=1, index=torch.tensor([0]), value=-1)
            tg_conf_gt_mask = torch.zeros((rgb_data.shape[0], data_list[0][2][3].shape[0], tg_max_obj), device=rgb_data.device)
            
            for idx in range(len(data_list)):
                tg_data_ = data_list[idx][2][2]
                tg_bbox_gt[idx][:, 0, :tg_data_.shape[-1]] = tg_data_[:, 0, :]
                tg_conf_gt_mask[idx][:, :tg_data_.shape[-1]] = data_list[idx][2][3]
            
        else:
            rgb_data    = torch.stack([data_[0] for data_ in data_list])
            
        xywh_mask   = torch.stack([data_[1][0] for data_ in data_list])
        cls_mask    = torch.stack([data_[1][1] for data_ in data_list])
        
        max_obj = [data_[1][2].shape[-1] for data_ in data_list]
        max_obj = max(max_obj)
        bbox_gt = torch.zeros((rgb_data.shape[0], 4, 1, max_obj), 
                              device=rgb_data.device).index_fill_(dim=1, index=torch.tensor([0]), value=-1)
        conf_gt_mask = torch.zeros((rgb_data.shape[0], data_list[0][1][3].shape[0], max_obj), device=rgb_data.device)
        
        for idx in range(len(data_list)):
            data_ = data_list[idx][1][2]
            bbox_gt[idx][:, 0, :data_.shape[-1]] = data_[:, 0, :]
            conf_gt_mask[idx][:, :data_.shape[-1]] = data_list[idx][1][3]
        
        if len(data_list[0][0]) == 2:
            return (rgb_data, tg_data), ((xywh_mask, bbox_gt, conf_gt_mask, cls_mask), (tg_xywh_mask, tg_bbox_gt, tg_conf_gt_mask, tg_cls_mask))
        else:
            return rgb_data, (xywh_mask, bbox_gt, conf_gt_mask, cls_mask)
    
    @staticmethod
    def val_bbox_collate(data_list):
        """
            This method creates the ground truth required for calculating the localization loss of the bounding boxes.

        Args:
            data_list (_type_): A list of (data, labels) with the RGB data for original YOWO
        """
        data = torch.stack([data_[0] for data_ in data_list])
        bbox_gt = [data[1] for data in data_list]       # The list of labels (not a tensor due to different number of objects per frame)
        
        if len(data_list[0]) == 3:      # For concat case with TG data
            tg_data = torch.stack([data_[2] for data_ in data_list])
            return (data, tg_data), bbox_gt

        return data, bbox_gt
    
    @staticmethod
    def test_bbox_collate(data_list):
        """
            The test data is the list of clips per video. That's why the output is also a list of lists for the videos
            in the batch.
        Args:
            data_list (_type_): _description_
        """

        clip_list = []
        gt_list = []
        frame_list = []
        tg_list = []
        
        for data in data_list:
            clip_list.append(torch.stack(data[0]))
            gt_list.append(data[1])
            frame_list.append(data[2])

            if len(data_list[0]) == 4:
                tg_list.append(torch.stack(data[3]))

        if tg_list:
            return list(zip(clip_list, tg_list)), gt_list, frame_list
            
        return clip_list, gt_list, frame_list

    @staticmethod
    def long_vid_collate(data_list):
        """
            The test data is the list of clips per video. That's why the output is also a list of lists for the videos
            in the batch.
        Args:
            data_list (_type_): _description_
        """
        clip_data = torch.stack([data[0] for data in data_list])
        frame_list = [data[1] for data in data_list]

        return clip_data, frame_list
