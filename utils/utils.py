"""
    Non-maximum suppression and batches Non-maximum suppression methods are implemented in this script.
"""
import torch
from torchvision.ops import distance_box_iou_loss
from typing import Tuple, Dict

def diou_nms(bbox_preds: torch.Tensor, 
             scores: torch.Tensor, 
             iou_threshold: float, 
             nms_thr_type: str='hard', 
             sorted=False) -> torch.Tensor:
    """
        This function implements the Non-maximum suppression by removing predictions that have
        an IoU greater than the iou_thr with another bbox with a higher confidence score.

    Args:
        bbox_preds (torch.Tensor): A tensor of boxes of the format (x1, y1, x2, y2) 
        scores (torch.Tensor): The corresponding confidence scores with the shape of (N,)
        iou_thr (float): The iou threshold
        nms_thr_type (str): Used to differentiate between normal NMS, Soft-NMS Linear, and Soft-NMS Gaussian.
    
    Returns:
        torch.Tensor: A tensor of the indices of the selected boxes.
    """
    if not sorted:
        
        sorted_idx = torch.argsort(scores, descending=True)
        bbox_preds = bbox_preds[sorted_idx]    

    tot_boxes = bbox_preds.shape[0]
    for idx in range(tot_boxes):
        
        if scores[idx] <= 0.2:
            continue

        box_coord = bbox_preds[idx, :4].reshape(1, -1)
        # iou = xyxy_iou(box_coord, bbox_preds[idx + 1:, :4])       # the loss for normal NMS
        iou = 1 - distance_box_iou_loss(box_coord.expand(tot_boxes - (idx + 1), -1), bbox_preds[idx + 1:, :4])

        if nms_thr_type == 'soft_L':
            scores[idx + 1:][iou > iou_threshold] *= 1 - iou[iou > iou_threshold]
        elif nms_thr_type == 'soft_G':
            scores[idx + 1:] *= torch.exp(-(1.0 / 0.5) * (iou ** 2))

    return torch.where(scores > 0.2)[0]

def xyxy_iou(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
        This function calculates the iou between two sets of bounding boxes based on their top left and
        bottom right points.

        If an input is only a single box (shape=(1, 4)), it will be broadcasted for IoU calculation, otherwise, the IoU 
        between the corresponding rows will be calculated.

    Args:
        bbox1 (torch.Tensor): A tensor of shape (N, 4) with the format of (x1, y1, x2, y2) where 0 <= x1 <= x2, and 0 <= y1 <= y2
        bbox2 (torch.Tensor): A tensor of shape (N, 4) with the format of (x1, y1, x2, y2) where 0 <= x1 <= x2, and 0 <= y1 <= y2

    Returns:
        torch.Tensor: A tensor of shape (N, 4) containing the resulting IoUs.
    """

    min_x = torch.minimum(bbox1[:, 0], bbox2[:, 0])
    max_x = torch.maximum(bbox1[:, 2], bbox2[:, 2])
    min_y = torch.minimum(bbox1[:, 1], bbox2[:, 1])
    max_y = torch.maximum(bbox1[:, 3], bbox2[:, 3])

    bbox1_w = bbox1[:, 2] - bbox1[:, 0]
    bbox1_h = bbox1[:, 3] - bbox1[:, 1]
    bbox2_w = bbox2[:, 2] - bbox2[:, 0]
    bbox2_h = bbox2[:, 3] - bbox2[:, 1]
    
    cw = bbox1_w + bbox2_w - (max_x - min_x)     # cw, ch, iou, ... shape: torch.Size([N])
    ch = bbox1_h + bbox2_h - (max_y - min_y)

    cw[cw <= 0] = 0
    ch[ch <= 0] = 0

    i_area = cw * ch
    u_area = bbox1_w * bbox1_h + bbox2_w * bbox2_h - i_area

    iou = i_area / u_area
    
    return iou

def pred_gt_iou(bbox_pred: torch.Tensor, bbox_gt: torch.Tensor, eps: float=1.0e-16) -> torch.Tensor:
    """
        This function calculates the IoU between the predicted bounding boxes and the ground truth objects.
        The reason for using a separate function is the difference in the tensor shapes.

    Args:
        bbox_pred (torch.Tensor): A tensor of shape (batch_size, pred_count, dim) with the the format of the first
        4 elements of the last dim being (x1, y1, x2, y2) where 0 <= x1 <= x2, and 0 <= y1 <= y2

        bbox_gt (torch.Tensor): A tesnor of shape (batch_size, 4, 1, num_obj) in which the 2nd dim is for 
        (c_x, c_y, w, h)

    Returns:
        torch.Tensor: A tensor of shape (batch_size, pred_count, num_obj) containing the calculated IoUs.
    """
    min_x = torch.minimum(bbox_pred[:, :, 0], bbox_gt[:, 0] - bbox_gt[:, 2]/2.0)
    min_y = torch.minimum(bbox_pred[:, :, 1], bbox_gt[:, 1] - bbox_gt[:, 3]/2.0)
    max_x = torch.maximum(bbox_pred[:, :, 2], bbox_gt[:, 0] + bbox_gt[:, 2]/2.0)
    max_y = torch.maximum(bbox_pred[:, :, 3], bbox_gt[:, 1] + bbox_gt[:, 3]/2.0)

    bbox_w = bbox_pred[:, :, 2] - bbox_pred[:, :, 0]
    bbox_h = bbox_pred[:, :, 3] - bbox_pred[:, :, 1]

    cw = bbox_w + bbox_gt[:, 2] - (max_x - min_x)
    ch = bbox_h + bbox_gt[:, 3] - (max_y - min_y)

    cw[cw <= 0] = 0
    ch[ch <= 0] = 0

    cw *= torch.sign(bbox_gt[:, 0])
    cw[cw <= 0] = 0

    i_area = cw * ch
    u_area = bbox_w * bbox_h + bbox_gt[:, 2] * bbox_gt[:, 3] - i_area

    iou = i_area / (u_area + eps)

    return iou

def cxcywh_to_xyxy(bbox: torch.Tensor, cfg: Dict, normalize: bool=False) -> torch.Tensor:
    """Converting the pred and label boxes from cxcywh format in the feature map dimension to 
    top left and bottom right format in the same dimension.

    Args:
        bbox (torch.Tensor): A tensor of shape (N, m > 4) or (N, nA*nH*nW, m > 4) with first 4
        columns in the cxcywh format in feature map coordinates.

        cfg (Dict): _description_

        normalize (bool): If True, the output xy coordinates will be in the [0, 1] range

    Returns:
        torch.Tensor: A tensor of shape (N, m > 4) with first 4 columns in the xyxy format in the absolute image size.
    """
    
    img_w   = cfg.localization_cfg.bbone_w
    img_h   = cfg.localization_cfg.bbone_h

    if bbox.dim() == 3:
        top_left_x      = (bbox[:, :, 0] - bbox[:, :, 2]/2.0).unsqueeze(dim=-1)
        top_left_y      = (bbox[:, :, 1] - bbox[:, :, 3]/2.0).unsqueeze(dim=-1)
        bottom_right_x  = (bbox[:, :, 0] + bbox[:, :, 2]/2.0).unsqueeze(dim=-1)
        bottom_right_y  = (bbox[:, :, 1] + bbox[:, :, 3]/2.0).unsqueeze(dim=-1)
    
    elif bbox.dim() == 2:
        top_left_x      = (bbox[:, 0] - bbox[:, 2]/2.0).unsqueeze(dim=-1)
        top_left_y      = (bbox[:, 1] - bbox[:, 3]/2.0).unsqueeze(dim=-1)
        bottom_right_x  = (bbox[:, 0] + bbox[:, 2]/2.0).unsqueeze(dim=-1)
        bottom_right_y  = (bbox[:, 1] + bbox[:, 3]/2.0).unsqueeze(dim=-1)

    # This should always be applied except during training which would reduce 
    # the loss on large errors. Thus, delaying convergence.
    top_left_x = torch.clamp(top_left_x, min=0, max=img_w)
    top_left_y = torch.clamp(top_left_y, min=0, max=img_h)
    bottom_right_x = torch.clamp(bottom_right_x, min=0, max=img_w)
    bottom_right_y = torch.clamp(bottom_right_y, min=0, max=img_h)

    assert (bottom_right_x >= top_left_x).all(), f"bad box: x1 larger than x2, {top_left_x.shape} {top_left_x[0]}"
    assert (bottom_right_y >= top_left_y).all(), "bad box: y1 larger than y2"

    if bbox.dim() == 3:
        bbox[:, :, :4] = torch.cat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], dim=-1)

        if normalize:
            bbox[:, :, 0] /= img_w
            bbox[:, :, 2] /= img_w
            bbox[:, :, 1] /= img_h
            bbox[:, :, 3] /= img_h
    elif bbox.dim() == 2:
        bbox[:, :4] = torch.cat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], dim=-1)

        if normalize:
            bbox[:, 0] /= img_w
            bbox[:, 2] /= img_w
            bbox[:, 1] /= img_h
            bbox[:, 3] /= img_h

    return bbox

def project_bbox_to_img_coord(bbox: torch.Tensor, cfg: Dict) -> torch.Tensor:
    """This method maps the predicted bbox from the feature map coordinates to the absolute image coordinates.

    Args:
        bbox (torch.Tensor): A tensor of shape (N, m>4), with the first 4 columns corresponding to xyxy bbox
        coordinates in the feature map.

    Returns:
        torch.Tensor: The input bbox in absolute image coordinates.
    """
    img_w   = cfg.data_prep.fr_width
    img_h   = cfg.data_prep.fr_height

    x_scale = img_w / cfg.localization_cfg.bbone_w
    y_scale = img_h / cfg.localization_cfg.bbone_h

    bbox[:, 0] *= x_scale
    bbox[:, 2] *= x_scale
    bbox[:, 1] *= y_scale
    bbox[:, 3] *= y_scale

    bbox[:, 0] = torch.clamp(bbox[:, 0], min=0, max=img_w - 1)
    bbox[:, 2] = torch.clamp(bbox[:, 2], min=0, max=img_w - 1)
    bbox[:, 1] = torch.clamp(bbox[:, 1], min=0, max=img_h - 1)
    bbox[:, 3] = torch.clamp(bbox[:, 3], min=0, max=img_h - 1)

    return bbox
