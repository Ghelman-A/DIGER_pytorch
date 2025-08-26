"""
    This class implements the per-frame inference metrics such as accuracy, localization, precision, etc.
    
    Author: Ali Ghelmani,                   Date: Dec. 2022
"""
import torch
import torch.nn as nn
import numpy as np
from utils.utils import pred_gt_iou, project_bbox_to_img_coord
from utils.pred_bbox import PredBbox
from typing import List, Tuple

class PerFrameMetrics(nn.Module):
    def __init__(self, cfg, is_test: bool=False) -> None:
        super().__init__()
        self.cfg = cfg
        self.l_cfg = cfg.localization_cfg
        self.is_test = is_test      # Confusion matrix are only needed during testing

        self.bbox_pred = PredBbox(cfg)
        
    def compute_metrics(self, model_output: torch.Tensor,
                        labels: List[torch.Tensor],
                        predict: bool=False) -> Tuple[Tuple[float], List[torch.Tensor], Tuple[List[int], List[int]]]:
        
        """This method computes the per-frame metrics for the localization and classification accuracy based on
        the bboxes with conf_score > 0.5.

        Args:
            model_output (torch.Tensor): Backbone output with the shape (N, nA*(4+1), nH, nW)
            labels (List[torch.Tensor]): List of labels for the output clips with the shape (5, 1, n_Obj)
            predict (bool): If predict is true, no need to calculate performance metrics

        Returns:
            Tuple[Tuple[float], List[torch.Tensor], Tuple[List[int], List[int]]]: 
                -   Calculated metrics, 
                -   A list of predicted bboxes for each clip (after nms) in the xyxy format in the absolute image dimensions,
                -   A list of class predictions and true labels to be used for confusion matrix generation. 
                    (The predictions are for matching bboxes only)
        """
        
        total_gt = 0
        total_detected = 0
        total_proposals = 0
        total_correct = 0
        conf_mat_pred = []
        conf_mat_label = []

        tot_trimmed_boxes = self.bbox_pred.pred_bbox_after_nms(model_output)

        if predict:
            tot_trimmed_boxes = [project_bbox_to_img_coord(trimmed_boxes, self.cfg) for trimmed_boxes in tot_trimmed_boxes]
            return (), tot_trimmed_boxes, ()

        for idx, trimmed_boxes in enumerate(tot_trimmed_boxes):
            total_proposals += trimmed_boxes.shape[0]
            total_gt += labels[idx].shape[-1]

            if trimmed_boxes.shape[0] == 0:         # No bbox with conf > thr!
                continue
            
            iou = pred_gt_iou(trimmed_boxes.unsqueeze(dim=0).unsqueeze(dim=-1), labels[idx].unsqueeze(dim=0)).squeeze(dim=0)
            best_iou, best_idx = torch.max(iou, dim=0)      

            for i, best_i in enumerate(best_idx):
                if best_iou[i] > 0.5:               # If IoU with GT > 0.5, the prediction is considered correct
                    total_detected += 1                    
                    
                    if trimmed_boxes[best_i, 5] == labels[idx][-1, 0, i]:
                        total_correct += 1
            
                    if self.is_test:
                        conf_mat_pred.append(trimmed_boxes[best_i, 5].int().item())
                        conf_mat_label.append(labels[idx][-1, 0, i].int().item())                
                        tot_trimmed_boxes[idx] = project_bbox_to_img_coord(trimmed_boxes, self.cfg)
        
        loc_precision = total_detected / (total_proposals + self.cfg.eps)
        cls_recall = total_correct / (total_gt + self.cfg.eps)
        classification_acc = total_correct / (total_detected + self.cfg.eps)
        loc_recall = total_detected / (total_gt + self.cfg.eps)
        loc_fscore = (2.0 * loc_precision * loc_recall) / (loc_precision + loc_recall + self.cfg.eps)

        return (loc_precision, cls_recall, loc_fscore, classification_acc, loc_recall), tot_trimmed_boxes, (conf_mat_pred, conf_mat_label)
