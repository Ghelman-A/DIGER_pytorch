import torch
import torch.nn.functional as F

import torch.nn as nn
from losses.focalloss import FocalLoss
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss
from utils.utils import pred_gt_iou, cxcywh_to_xyxy
from utils.pred_bbox import PredBbox
from typing import Tuple, Union

class RegionLoss_DIGER(nn.Module):
    """
        This class implements the object localization loss

    Args:
        nn (_type_): _description_
    """
    def __init__(self, cfg):
        super(RegionLoss_DIGER, self).__init__()
        self.cfg = cfg
        self.l_cfg = cfg.localization_cfg
        self.tg_cfg = self.cfg.arch_cfg.tg_cfg
        self.cl_cfg = self.tg_cfg.cl_cfg
        self.bbox_pred_size = 4 + 1 + 3 #Todo: add a num_classes variable to cfg

        self.focal_loss = FocalLoss(class_num=3)
        self.bbox_pred = PredBbox(cfg)               

    def build_conf_mask(self, bbox_pred: torch.Tensor, bbox_gt: torch.Tensor, conf_gt_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            This method produces a mask for the confidence score of the predicted bounding boxes.
            There are 3 mechanism involved:

                1. To have as low of conf score as possible for areas without an object, there is a loss 
                   with no_obj_scale coefficient.
                
                2. If the IoU of a predicted bbox with a ground truth is higher than a threshold, it is assumed
                   to contain an object and thus there is no loss for it.
                   
                3. The last one is the loss for the conf score of the best anchor box, which is expected to be
                   high and detect the object. 
                   
                   The difference between this loss and the No. 2 is that in 2, it is
                   possible to have a big bbox that contains an object but it's not the best anchor box. while
                   there is no loss for such cases, we still expect the best anchor to predict the object.       

        Args:
            bbox_coord (_type_): (cent_x, cent_y, w, h) coordinates of the bounding boxes used to calculate IoU.
            conf (_type_): the conf scores of the input batch for which a mask if to be generated.
        """

        bbox_pred_ = self.bbox_pred.output_bbox(bbox_pred)

        iou = pred_gt_iou(bbox_pred_.unsqueeze(dim=-1), bbox_gt)
        gt_iou = iou * conf_gt_mask

        iou = torch.amax(iou, dim=-1)
        gt_iou = torch.amax(gt_iou, dim=-1)
        conf_gt_mask = torch.amax(conf_gt_mask, dim=-1)

        conf_mask = torch.ones_like(iou) * self.l_cfg.no_obj_scale
        conf_mask[iou > self.cfg.localization_cfg.obj_thr] = 0
        conf_mask[conf_gt_mask > 0] = self.l_cfg.obj_scale

        return conf_mask, gt_iou

    def forward(self, output, label: Tuple[torch.Tensor], current_epoch: int=None) -> int:
        distill_mode = False
        
        if isinstance(output, tuple):
            distill_mode = True
            output, kd_loss = self.kd_loss(output, current_epoch)
            
        out_sh = output.shape
        xywh_mask, bbox_gt, conf_gt_mask, cls_mask = label

        new_out = output.reshape(out_sh[0], self.l_cfg.num_anchors, self.bbox_pred_size, out_sh[-2], out_sh[-1])
        new_out = new_out.permute(0, 1, 3, 4, 2).reshape(output.shape[0], -1, self.bbox_pred_size)      # Size (nBatch, nA*nW*nH, (4 + 1 + 3))
        new_out[:, :, :2] = torch.sigmoid(new_out[:, :, :2])                # Converting the output x, y into [0, 1] range

        xywh_output = new_out[:, :, :4]
        cls_output  = new_out[:, :, 5:]
        conf_output = torch.sigmoid(new_out[:, :, 4])                       # Converting the output conf into [0, 1] range
        
        conf_mask, conf_gt = self.build_conf_mask(xywh_output.clone().detach(), bbox_gt, conf_gt_mask)

        if self.cfg.arch_cfg.loss == 'smooth_l1':
            xywh_loss = nn.SmoothL1Loss(reduction='mean')(xywh_output[xywh_mask>-10], xywh_mask[xywh_mask>-10]) * 20.0
        
        elif self.cfg.arch_cfg.loss in ['ciou', 'diou']:
            gt_mask = self.prep_ciou_gt(bbox_gt, conf_gt_mask)            
            xywh_output = self.bbox_pred.output_bbox(xywh_output, normalize=True)         # converting to xyxy format in feature map coord
            
            xywh_output = xywh_output[gt_mask > 0].reshape(-1, 4)

            gt = gt_mask[gt_mask > 0].reshape(-1, 4)
            gt = cxcywh_to_xyxy(gt, self.cfg, normalize=True)

            if self.cfg.arch_cfg.loss == 'ciou':
                xywh_loss = complete_box_iou_loss(xywh_output, gt, reduction='mean') * 15.0
            elif self.cfg.arch_cfg.loss == 'diou':
                xywh_loss = distance_box_iou_loss(xywh_output, gt, reduction='mean') * 15.0
            
        conf_loss = nn.MSELoss(reduction='mean')(conf_output * conf_mask, conf_gt * conf_mask) * 10.0
        loss_cls = self.l_cfg.cls_scale * self.focal_loss(cls_output, cls_mask) * 1.5

        # sum of loss
        loss = xywh_loss + conf_loss + loss_cls
        if distill_mode: loss = loss + kd_loss

        return loss

    def kd_loss(self, output: Union[torch.Tensor, Tuple[torch.Tensor, ...]], current_epoch: int) -> Tuple[torch.Tensor, int]:
        kd_cfg = self.tg_cfg.kd_loss

        if self.cl_cfg.apply_cl and current_epoch is not None:
            scale_2d = 0 if current_epoch < self.cl_cfg._2d.start_epoch else self.cl_cfg._2d.kd_scale
            scale_3d = 0 if current_epoch < self.cl_cfg._3d.start_epoch else self.cl_cfg._3d.kd_scale
        else:
            scale_2d = self.cl_cfg._2d.kd_scale
            scale_3d = self.cl_cfg._3d.kd_scale

        #------------ For the aux_distill and distill cases -------------#
        kd_loss_tot = 0
        for kd_out in output[1]:
            if isinstance(kd_out, tuple):
                distill_3d_output = kd_out[0] # .clone()
                distill_2d_output = kd_out[1] # .clone()

                distill_3d_output = distill_3d_output / torch.linalg.norm(distill_3d_output, dim=1, keepdims=True, ord=2)
                distill_2d_output = distill_2d_output / torch.linalg.norm(distill_2d_output, dim=1, keepdims=True, ord=2)

                if self.tg_cfg.kd_loss.loss == 'l1_loss':
                    kd_loss = scale_3d * torch.linalg.norm(distill_3d_output[:, :, 0] - 
                                                           distill_3d_output[:, :, 1], ord=1, dim=-1).abs().mean()
                    kd_loss = kd_loss + scale_2d * torch.linalg.norm(distill_2d_output[:, :, 0] - 
                                                                     distill_2d_output[:, :, 1], ord=1, dim=-1).abs().mean()
                
                elif self.tg_cfg.kd_loss.loss == 'l2_loss':
                    kd_loss = scale_3d * torch.linalg.norm(distill_3d_output[:, :, 0] - 
                                                           distill_3d_output[:, :, 1], ord=2, dim=-1).abs().mean()
                    kd_loss = kd_loss + scale_2d * torch.linalg.norm(distill_2d_output[:, :, 0] - 
                                                                     distill_2d_output[:, :, 1], ord=2, dim=-1).abs().mean()
                
                elif self.tg_cfg.kd_loss.loss == 'cosine_similarity':
                    kd_loss = scale_3d * F.cosine_similarity(distill_3d_output[:, :, 0], 
                                                             distill_3d_output[:, :, 1], dim=-1).abs().mean()
                    kd_loss = kd_loss + scale_2d * F.cosine_similarity(distill_2d_output[:, :, 0], 
                                                                       distill_2d_output[:, :, 1], dim=-1).abs().mean()

                elif self.tg_cfg.kd_loss.loss == 'cross_entropy':
                    kd_loss = scale_3d * self.custom_cross_entropy(distill_3d_output[:, :, 0], 
                                                                   distill_3d_output[:, :, 1],
                                                                   kd_cfg.temp_3d_rgb, kd_cfg.temp_3d_tg)
                    kd_loss = kd_loss + scale_2d * self.custom_cross_entropy(distill_2d_output[:, :, 0], 
                                                                             distill_2d_output[:, :, 1],
                                                                             kd_cfg.temp_2d_rgb, kd_cfg.temp_2d_tg)
                
            else:
                distill_output = kd_out # .clone()
                distill_output = distill_output / torch.linalg.norm(distill_output, dim=1, keepdims=True, ord=2)
                
                if self.tg_cfg.kd_loss.loss == 'l1_loss':
                    kd_loss = scale_3d * torch.linalg.norm(distill_output[:, :, 0] - distill_output[:, :, 1], ord=1, dim=-1).abs().mean()
                
                elif self.tg_cfg.kd_loss.loss == 'l2_loss':
                    kd_loss = scale_3d * torch.linalg.norm(distill_output[:, :, 0] - distill_output[:, :, 1], ord=2, dim=-1).abs().mean()
                
                elif self.tg_cfg.kd_loss.loss == 'cosine_similarity':
                    kd_loss = scale_3d * F.cosine_similarity(distill_output[:, :, 0], distill_output[:, :, 1], dim=-1).abs().mean()
                
                elif self.tg_cfg.kd_loss.loss == 'cross_entropy':
                    kd_loss = scale_3d * self.custom_cross_entropy(distill_output[:, :, 0], distill_output[:, :, 1],
                                                                   kd_cfg.temp_3d_rgb, kd_cfg.temp_3d_tg)
            
            kd_loss_tot = kd_loss_tot + kd_loss
            
        output = output[0]

        return output, kd_loss_tot

    def prep_ciou_gt(self, bbox_gt: torch.Tensor, conf_gt_mask: torch.Tensor) -> torch.Tensor:
                
        # Original bbox shape: (N, 4, 1, n_obj), After change: (N, 245, 4, n_obj), 245=(nA*nW*nH)
        new_gt = bbox_gt.squeeze(dim=2).unsqueeze(dim=1).expand(-1, conf_gt_mask.shape[1], -1, -1)
        
        # Original conf_gt_mask shape: (N, 245, n_obj)
        new_gt = new_gt * conf_gt_mask.unsqueeze(dim=2)
        new_gt = torch.amax(new_gt, dim=-1)         # Final shape: (N, 245, 4)
        
        return new_gt

    @staticmethod
    def custom_cross_entropy(output: torch.Tensor, target: torch.Tensor, out_temp: float=1.0, target_temp: float=1.0) -> torch.Tensor:
        """A custom cross_entropy function return due to uncertainty in the use of the PyTorch version.

        Args:
            output (torch.Tensor): _description_
            target (torch.Tensor): _description_
            out_temp (float, optional): model output temperature parameter. Defaults to 1.0.
            target_temp (float, optional): target temperature parameter. Defaults to 1.0.

        Returns:
            torch.Tensor: The calculated loss.
        """
        log_p = F.log_softmax(output / out_temp, dim=1)
        log_q = F.log_softmax(target / target_temp, dim=1)

        loss = -(torch.exp(log_q) * log_p).sum(dim=1).mean()
        return loss
