import pandas as pd
import numpy as np
import torch
import os

from pathlib import Path
import pickle
import xml.etree.ElementTree as ET
import math
from typing import List


class PrepareDataLists:
    """
        This class prepares the labeled data for train, validation, and test cases.
        
        The evaluation and test datasets are prepared per clip and not per video due to the 
        needed per-frame metrics.
    """
    def __init__(self, config):
        self.cfg = config
        self.tg_cfg = self.cfg.arch_cfg.tg_cfg
        self.prep_cfg = config.data_prep
        self.tr_mode_cfg = self.cfg[self.cfg.train_mode]
        self.classes = config.OBJECT_ACTION_NAMES.excavator

        self.train_csv = config.data_prep.train_csv_dir
        self.eval_csv = config.data_prep.eval_csv_dir
        self.test_csv = config.data_prep.test_csv_dir

        self.save_dir = config.ds_list_dir
        with open(config.ds_hdf5_idx_dir, 'rb') as f:
            self.dset_idx = pickle.load(f)
        
        self.anchors = np.loadtxt(config.localization_cfg.anchor_save_dir, delimiter=', ')
        self.anchors[:, 0] *= config.localization_cfg.bbone_w          # the saved anchors are normalized
        self.anchors[:, 1] *= config.localization_cfg.bbone_h

    def train(self):
        """
            This class creates DataFrame for supervised train dataset along with its corresponding save file name,
            and passes the result as arguments to the __prep_data__ method for final dataset creation and saving.
        :return: None
        """
        save_dir = os.path.join(self.save_dir, f'tr_{self.tg_cfg.skip_step}')
        os.makedirs(save_dir, exist_ok=True)
        
        train_df = pd.read_excel(self.train_csv, index_col=0)
        eval_df = pd.read_excel(self.eval_csv, index_col=0)
        
        self.__prep_data__(train_df, save_dir, 'train', skip_step=self.tr_mode_cfg.skip_step)
        self.__prep_data__(eval_df, save_dir, 'eval', skip_step=self.tr_mode_cfg.eval_skip, train=False)

    def test(self):
        """
            !Note: This method is deprecated and the predict method is to be used in its place instead due to the
            video-level clip extraction mechanism.
            This method prepares the test data. Due to the frame-level metrics needed for testing, this method
            uses the same approach as train and eval for data preparation.
        """        
        save_dir = os.path.join(self.save_dir, 'supervised_TG')
        os.makedirs(save_dir, exist_ok=True)

        test_df = pd.read_excel(self.test_csv, index_col=0)
        self.__prep_data__(test_df, save_dir, 'test', train=False)

    def predict(self):
        """This method is used to create predict dataset which at this stage is comprised of eval_pred and test_pred
        datasets. The reason for creating these separate datasets is the requirement for assessing the performance of
        model on a per video basis.
        """
        save_dir = os.path.join(self.save_dir, f'cyclic_supervised_TG_skip_{self.tg_cfg.skip_step}')
        os.makedirs(save_dir, exist_ok=True)

        eval_df = pd.read_excel(self.eval_csv, index_col=0)
        test_df = pd.read_excel(self.test_csv, index_col=0)

        self.__per_vid_data__(eval_df, save_dir, 'eval')
        self.__per_vid_data__(test_df, save_dir, 'test')
    
    def __prep_data__(self, train_df, save_dir, name_prefix, clip_len=16, skip_step=2, train=True):
        """
        \\UPDATE: The cyclic part was removed due to a problem with the Albumentation cropping!

        This method extracts the clips from each video. Except for the skipped frames, all of the frames are used
        for clip generation and each frame is set once, as the end frame of one clip.
        
        NOTE: All of the extracted clips are treated as separate data samples on which the model can
        be trained.

            ####### output format in each row of the dataframe: [idx, GroundTruth, labels, frame_names]
            The idx is from the index of the selected frame in the h5py dataset. In other words, this function only
            selects the index of the train data or the index of the frames in each clip, and the final data are queried
            from the h5py file during training in the Dataset class. (i.e., the get_item() function)

        :param train_df: The input dataframe with information about the dataset, used to extract clips.
        :param name_prefix: A way to handle different cases of train, validation, and test.
        :param clip_len:
        :param skip_step:
        :return:
        """

        save_file = os.path.join(save_dir, f'{name_prefix}_{clip_len}_skip_{skip_step}.pkl')
        headers = self.prep_cfg.raw_dataset_csv_headers

        #------------------------------------------------------------------------------------#
        # Creates clips from the input video based on the given skip step, clip len, and     #
        # frame overlap, and saves the name of the selected frames in a dataframe for use    #
        # during training.                                                                   #
        #------------------------------------------------------------------------------------#
        final_frames = []

        for _, data in train_df.iterrows():
            """
                The frames are selected based on the skip step. For example if skip step is 3 we have:
                    - 0, 3, 6, 9, 12, ...
                
                To build the clip with the required number of frames for the starting frames, the first
                frame (frame[0]) is repeated as needed.
            """
            curr = Path(data[headers[1]])
            frame_list = sorted(list(curr.glob("*.PNG")))[::skip_step]
            label_list = sorted(list(curr.glob("*.xml")))[::skip_step]

            if (len(frame_list) / self.tg_cfg.skip_step) < clip_len + 1:
                print(frame_list[0].name)
                continue

            for frame_idx in range(0, len(frame_list)):         # range(start, end, step)
                clip = []

                idx = np.arange(frame_idx, frame_idx - clip_len - 1, -1)
                idx[idx < 0] = 0    # Repeating the first frame
                
                clip = sorted(idx)                
                fr_names = [str(frame_list[clip_]) for clip_ in clip]
                
                dset_idx = [self.dset_idx[fr] for fr in fr_names]

                annot_list = [self.get_annot_info(str(label_list[clip_])) for clip_ in clip]
                # gt_list = self.build_targets(annot_list) if train else self.build_test_targets(annot_list)
                final_frames.append((dset_idx, annot_list, fr_names))

        data_df = pd.DataFrame(final_frames, columns=['data', 'label', 'frame_names'])
        print(f"Length of {save_file}: {len(data_df)}")

        ##### Writing the dataset into a pickle file
        with open(save_file, 'wb') as f:
            pickle.dump(data_df, f)

    def __per_vid_data__(self, data_df: pd.DataFrame, save_dir: str, name_prefix: str, clip_len: int=16, skip_step: int=2) -> None:
        """This method creates clips of the given length from all of the frames of each video in the given dataset.
        The difference with the previous method is that in this case the clips are created in a per video basis and
        are not all bundeled together with clips of other videos.
        
        NOTE: The output data format is thus a list of clips for each video, the corresponding ground truthes for each clip,
        and finally, the names of all of the frames in the video as a list.
        
        Output format:
            [
                vid_0: [[clip_0, label(x, y, w, h), frame_list], [clip_1, label(x, y, w, h), frame_list], ....],
                vid_1: [[clip_0, label(x, y, w, h), frame_list], [clip_1, label(x, y, w, h), frame_list], ....],
                ...
            ]

        Args:
            data_df (pd.DataFrame): Dataframe containing the list of the videos used for train, val, or test
            save_dir (str): _description_
            name_prefix (str): Used to specify if the videos are for validation or test
            clip_len (int, optional): _description_. Defaults to 16.
            skip_step (int, optional): _description_. Defaults to 2.
        """

        save_file = os.path.join(save_dir, f'predict_{name_prefix}_clip_len_{clip_len}_skip_{skip_step}.pkl')
        headers = self.prep_cfg.raw_dataset_csv_headers

        #------------------------------------------------------------------------------------#
        # Creates clips from the input video based on the given skip step, clip len, and     #
        # frame overlap, and saves the name of the selected frames in a dataframe for use    #
        # during training.                                                                   #
        #------------------------------------------------------------------------------------#
        final_frames = []

        for _, data in data_df.iterrows():
            """
                The frames are selected based on the skip step. For example if skip step is 3 we have:
                    - 0, 3, 6, 9, 12, ...
                
                To build the clip with the required number of frames for the starting frames, the first
                frame (frame[0]) is repeated as needed.
            """
            vid_clips = []
            vid_gts = []
            curr = Path(data[headers[1]])
            frame_list = sorted(list(curr.glob("*.PNG")))[::skip_step]
            label_list = sorted(list(curr.glob("*.xml")))[::skip_step]

            if self.cfg.data_prep.cyclic and len(frame_list) < clip_len:
                print(frame_list[0].name)
                continue

            for frame_idx in range(0, len(frame_list)):         # range(start, end, step)
                clip = []

                idx = np.arange(frame_idx, frame_idx - clip_len, -1)                
                if not self.cfg.data_prep.cyclic: idx[idx < 0] = 0    # Repeating the first frame

                clip = sorted(idx)
                fr_names = [str(frame_list[clip_]) for clip_ in clip]
                
                dset_idx = [self.dset_idx[fr] for fr in fr_names]
                gt_list = self.build_test_targets(self.get_annot_info(str(label_list[frame_idx])))

                vid_clips.append(dset_idx)
                vid_gts.append(gt_list)
            
            final_frames.append((vid_clips, vid_gts, frame_list))

        dset_df = pd.DataFrame(final_frames, columns=['data', 'built_GTs', 'frame_names'])
        print(f"Length of {save_file}: {len(dset_df)}")

        ##### Writing the dataset into a pickle file
        with open(save_file, 'wb') as f:
            pickle.dump(dset_df, f)

    def get_annot_info(self, file_name):
        """
            This function parses the xml label files and extracts the activity class and bounding box information.

        Args:
            file_name (_type_): The path of the xml label file

        Returns:
            list of lists: The extracted bbox information are in the center, width, and height format (c_x, c_y, w, h).
        """
        tr = ET.parse(file_name)
        root = tr.getroot()
        
        annot_info = []
        for obj in root.findall('object'):
            act_cls = self.classes.index(obj.find('name').text.lower())
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            center_x_norm = (xmin + xmax) / (2.0 * self.cfg.data_prep.fr_width)
            center_y_norm = (ymin + ymax) / (2.0 * self.cfg.data_prep.fr_height)
            bbox_w_norm = (xmax - xmin) / self.cfg.data_prep.fr_width
            bbox_h_norm = (ymax - ymin) / self.cfg.data_prep.fr_height

            annot_info.append([act_cls, center_x_norm, center_y_norm, bbox_w_norm, bbox_h_norm])
        
        return annot_info

    def build_targets(self, annot_list):
        """
            This method finds the best matching anchor box with the ground truth bbox and calculates
            the required training labels.
            
            This is done to speed up the training, however, it should be repeated every time the anchor
            boxes, the dataset, or the size of the final feature map is changed.

        Args:
            annot_list (list): List of the corresponding annotations for the selected frames.
        """
        #--------------------------------------------------------------------#
        #    Finding the best matching anchor for each ground truth bbox     #
        #--------------------------------------------------------------------#
        map_w = self.cfg.localization_cfg.bbone_w           # output feature map frame size
        map_h = self.cfg.localization_cfg.bbone_h
        num_anchors = self.cfg.localization_cfg.num_anchors
        num_cls = 3

        xywh_mask = -10.0 * torch.ones(size=(self.cfg.localization_cfg.num_anchors * 4, map_h, map_w), dtype=float)
        cls_mask = torch.zeros(size=(self.cfg.localization_cfg.num_anchors * 3, map_h, map_w), dtype=float)
        bbox_gt = torch.zeros(size=(4, 1, len(annot_list)))
        conf_gt_mask = torch.zeros(size=(len(annot_list), num_anchors, map_h, map_w))
        
        for idx, annot in enumerate(annot_list):
            gt_c_x      = annot[1] * map_w
            gt_c_y      = annot[2] * map_h
            gt_bbox_w   = annot[3] * map_w
            gt_bbox_h   = annot[4] * map_h

            best_anchor = self.get_best_anchor((gt_bbox_w, gt_bbox_h))

            gt_c_i = int(gt_c_x)
            gt_c_j = int(gt_c_y)
            
            target_x = gt_c_x - gt_c_i              # Pred output of YOLO: sigmoid(pred_x) + gt_c_i, (that's why target_x is compared with sigmoid(pred_x))
            target_y = gt_c_y - gt_c_j              # Same as for x above
            target_w = math.log(gt_bbox_w / self.anchors[best_anchor][0])       # Pred output of YOLO: b_w = anchor_w * exp(pred_w)
            target_h = math.log(gt_bbox_h / self.anchors[best_anchor][1])       # Pred output of YOLO: b_h = anchor_h * exp(pred_h)

            xywh_mask[best_anchor * 4:best_anchor * 4 + 4, gt_c_j, gt_c_i] = torch.tensor([target_x, target_y, target_w, target_h])
            bbox_gt[:, 0, idx] = torch.tensor([gt_c_x, gt_c_y, gt_bbox_w, gt_bbox_h])
            conf_gt_mask[idx, best_anchor, gt_c_j, gt_c_i] = 1
            
            # Making the one hot class labels
            label = torch.zeros(num_cls)
            label[annot[0]] = 1
            cls_mask[best_anchor * num_cls:best_anchor * num_cls + num_cls, gt_c_j, gt_c_i] = label

        #-----------------------------------------------------------------------#
        #           Changing the format of targets into (5*7*7, -1)             #
        #-----------------------------------------------------------------------#
        xywh_mask = xywh_mask.reshape(num_anchors, 4, map_h, map_w).permute(0, 2, 3, 1).reshape(-1, 4)
        cls_mask = cls_mask.reshape(num_anchors, num_cls, map_h, map_w).permute(0, 2, 3, 1).reshape(-1, num_cls)
        conf_gt_mask = conf_gt_mask.reshape(len(annot_list), -1).T
        
        return xywh_mask, cls_mask, bbox_gt, conf_gt_mask

    def build_test_targets(self, annot_list: List) -> torch.Tensor:
        
        #--------------------------------------------------------------------#
        #    Finding the best matching anchor for each ground truth bbox     #
        #--------------------------------------------------------------------#
        map_w = self.cfg.localization_cfg.bbone_w           # output feature map frame size
        map_h = self.cfg.localization_cfg.bbone_h

        bbox_gt = torch.zeros(size=(5, 1, len(annot_list)))
        
        for idx, annot in enumerate(annot_list):
            gt_c_x      = annot[1] * map_w
            gt_c_y      = annot[2] * map_h
            gt_bbox_w   = annot[3] * map_w
            gt_bbox_h   = annot[4] * map_h

            bbox_gt[:, 0, idx] = torch.tensor([gt_c_x, gt_c_y, gt_bbox_w, gt_bbox_h, annot[0]])
        
        return bbox_gt

    def get_best_anchor(self, gt_box):
        """
            This method searches through the anchor boxes to find the best matching one based on the IoU.

        Args:
            gt_box (list): width and height of the ground truth bbox.
        """
        gt_w = gt_box[0]
        gt_h = gt_box[1]
        gt_area = gt_w * gt_h
        best_anchor = 0
        best_iou = 0
        
        for idx, anchor in enumerate(self.anchors):
            cw = min(gt_w, anchor[0])
            ch = min(gt_h, anchor[1])
            
            i_area = cw * ch
            u_area = anchor[0] * anchor[1] + gt_area - i_area
            iou = i_area / u_area

            if iou > best_iou:
                best_anchor = idx
                best_iou = iou
        
        return best_anchor
