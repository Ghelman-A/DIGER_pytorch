"""
    Author: Ali Ghelmani,           Data: Dec. 2022
"""
import numpy as np
import torch
import torch.nn.functional as func
from torchvision.ops import batched_nms, nms
from utils.utils import diou_nms
from utils.utils import cxcywh_to_xyxy
from typing import List


class PredBbox:
    """
        This class converts the model outpus prediction to the desired format for the final Bbox
        dimensions using the YOLOv2 setup.
    """
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.l_cfg = cfg.localization_cfg
        self.bbox_pred_size = 4 + 1 + 3

        #-------------------------------------------------------------------------------------#
        #        The anchor values used for calculating the IoU with the ground truth.        #
        #-------------------------------------------------------------------------------------#
        self.anchors = np.loadtxt(cfg.localization_cfg.anchor_save_dir, delimiter=', ')
        self.anchors[:, 0] *= cfg.localization_cfg.bbone_w          # the saved anchors are normalized
        self.anchors[:, 1] *= cfg.localization_cfg.bbone_h

        self.anchors = torch.from_numpy(self.anchors)
        anchors = self.anchors.reshape(self.l_cfg.num_anchors, 2, 1, 1).expand(self.l_cfg.num_anchors, 2, self.l_cfg.bbone_h, self.l_cfg.bbone_w)
        self.anchors_w = anchors[:, 0].reshape(-1)
        self.anchors_h = anchors[:, 1].reshape(-1)

        x_range = torch.linspace(start=0, end=self.l_cfg.bbone_w - 1, steps=self.l_cfg.bbone_w)
        y_range = torch.linspace(start=0, end=self.l_cfg.bbone_h - 1, steps=self.l_cfg.bbone_h)
        x_range, y_range = torch.meshgrid(x_range, y_range, indexing='xy')
        
        self.grid_x = x_range.reshape(-1).repeat(5)
        self.grid_y = y_range.reshape(-1).repeat(5)

    def output_bbox(self, model_output: torch.Tensor, reshape_needed: bool=False, normalize: bool=False) -> List[torch.Tensor]:
        """This method converts the model output into the correct bbox predictions based on the
        equations described in the YOLOv2 paper.
        
        EQUATIONS:
                    - b_x = \sigma(model_out_center_x) + grid_x
                    - b_y = \sigma(model_out_center_y) + grid_y
                    - b_w = anchor_w * exp(model_out_w)
                    - b_h = anchor_h * exp(model_out_h)

        Args:
            model_output (torch.Tensor): Model output tensor with the shape of (N, nA*(4+1+n_cls), nH, nW) if reshape
            is needed and the shape of (N, nA*nH*nW, 4+1+n_cls) if already reshaped.

            reshape_needed (bool, optional): _description_. Defaults to False.
            normalize (bool, optional): If True, the output xy points will be in the [0, 1] range. Defaults to False.

        Returns:
            torch.Tensor: A tensor of shape (N, nA*nH*nW, 4+1+n_cls) with the correct bbox values.
        """
        
        if reshape_needed:
            out_sh = model_output.shape
            
            model_output = model_output.reshape(out_sh[0], self.l_cfg.num_anchors, self.bbox_pred_size, out_sh[-2], out_sh[-1])
            model_output = model_output.permute(0, 1, 3, 4, 2).reshape(model_output.shape[0], -1, self.bbox_pred_size)      # Size (nBatch, nA*nW*nH, (4 + 1 + 3))

            model_output[:, :, :2] = torch.sigmoid(model_output[:, :, :2])                # Converting the output x, y into [0, 1] range
        
        grid_x = self.grid_x.to(model_output.device)
        grid_y = self.grid_y.to(model_output.device)
        anchors_w = self.anchors_w.to(model_output.device)
        anchors_h = self.anchors_h.to(model_output.device)

        # Converting the output x, y coordinates
        model_output[:, :, 0] += grid_x
        model_output[:, :, 1] += grid_y
        
        # Preparing the w, h according to YOLOv2 format for IoU calculation
        model_output[:, :, 2] = torch.exp(model_output[:, :, 2]) * anchors_w
        model_output[:, :, 3] = torch.exp(model_output[:, :, 3]) * anchors_h

        model_output[:, :, :4] = cxcywh_to_xyxy(model_output[:, :, :4], self.cfg, normalize)

        return model_output
    
    def pred_bbox_after_nms(self, model_output: torch.Tensor) -> List[torch.Tensor]:
        """This method receives the model output in the shape of (N, nA*(4+1+n_cls), nH, nW) and produces
        the final bbox predictions after applying non-maxumum suppression.

        Args:
            model_output (torch.Tensor): Tensor of shape (N, nA*(4+1+n_cls), nH, nW)

        Returns:
            List[torch.Tensor]: A list of final output bbox predictions per frame in a Tensor of shape (N_pred, 7)
            in which the column data are (c_x, c_y, w, h, conf_score, max_cls_idx, max_cls_prob).
        """
        
        tot_trimmed_boxes = []
        pred_boxes = self.evaluate_pred_boxes(model_output)

        for idx in range(pred_boxes.shape[0]):
            frame_boxes = pred_boxes[idx]
            
            conf_idx = torch.argsort(frame_boxes[:, 4], descending=True)
            frame_boxes = frame_boxes[conf_idx]

            trimmed_boxes = self.batch_nms(frame_boxes)
            trimmed_boxes = trimmed_boxes[trimmed_boxes[:, 4] > self.l_cfg.pre_nms_thr]

            tot_trimmed_boxes.append(trimmed_boxes)
        
        return tot_trimmed_boxes

    def evaluate_pred_boxes(self, output: torch.Tensor) -> torch.Tensor:
        """This method simply applies a preliminary conf_score thresholding on the predicted bboxes.

        Args:
            output (torch.Tensor): A tensor of shape (N, nA*nH*nW, 4+1+n_cls) with bbox in the xyxy format

        Returns:
            torch.Tensor: Same as input after trimming the outputs with the low conf_score.
        """
        
        new_out = self.output_bbox(output, reshape_needed=True)
        
        # Converting the conf output
        new_out[:, :, 4] = torch.sigmoid(new_out[:, :, 4])
        max_cls_prob, max_cls_ind = torch.max(func.softmax(new_out[:, :, 5:], dim=-1), dim=-1)

        conf = new_out[:, :, 4] * max_cls_prob
        conf_ind = (conf > self.l_cfg.conf_thr).int()

        output_boxes = new_out[:, :, :7] * conf_ind.unsqueeze(dim=-1)
        output_boxes[:, :, 5] = max_cls_ind * conf_ind
        output_boxes[:, :, 6] = max_cls_prob * conf_ind
    
        #------------------------------------------------------------------------------------#
        #     output data shape: (c_x, c_y, w, h, conf_score, max_cls_idx, max_cls_prob)     #
        #                                                                                    #
        # It should be noted that the bbox coord are normalized w.r.t. the feature map size  #
        # which is similar to the format of the ground truth coord data.                     #
        #------------------------------------------------------------------------------------#
        return output_boxes

    def batch_nms(self, fr_boxes: torch.Tensor) -> torch.Tensor:
        """This method applies the selected NMS function to the predicted bounding boxes.

        Args:
            fr_boxes (torch.Tensor): The input predicted bboxes with shape (N, m>=5). Where the first
            4 columns are the bbox coordinates in the xyxy format, and the 5th column is the conf
            score.

        Returns:
            torch.Tensor: The final selected bounding boxes.
        """
        if self.cfg.arch_cfg.nms == 'nms':
            nms_out_idxs = nms(fr_boxes[:, :4], scores=fr_boxes[:, 4], iou_threshold=self.l_cfg.nms_thr)
        
        elif self.cfg.arch_cfg.nms == 'batched_nms':
            nms_out_idxs = batched_nms(fr_boxes[:, :4], scores=fr_boxes[:, 4], idxs=fr_boxes[:, 5], iou_threshold=self.l_cfg.nms_thr)
        
        elif self.cfg.arch_cfg.nms == 'diou_nms':
            nms_out_idxs = diou_nms(fr_boxes[:, :4], scores=fr_boxes[:, 4], iou_threshold=self.l_cfg.nms_thr,
                                    nms_thr_type=self.cfg.arch_cfg.nms_thr_type)
        
        return fr_boxes[nms_out_idxs]
