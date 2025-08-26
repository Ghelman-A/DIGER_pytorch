from __future__ import print_function, division

import cv2
import torch
from torch.utils.data import Dataset
from preprocessing.dataprocessing import PrepareDataLists
import pickle
import h5py
from typing import Dict, List
import matplotlib.pyplot as plt

class SupervisedDataset(Dataset):
    
    def __init__(self, cfg: Dict, pkl_file: str, transform, mode: str='train'):
        self.cfg = cfg
        self.tg_mode = bool(cfg.arch_cfg.tg_cfg.type)       # If type == '' no TG data is used
        self.transform = transform
        self.mode = mode
        self.prep_targets = PrepareDataLists(cfg)

        with open(pkl_file, 'rb') as f:
            self.dataset_df = pickle.load(f)

        if cfg.train_cfg.load_ds_to_mem:
            f = h5py.File(self.cfg.ds_hdf5_dir, 'r')
            self.loaded_ds = f['dset']

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):

        if self.mode == 'train':
            data = self.load_clip(self.dataset_df['data'][idx])

            label = self.dataset_df['label'][idx]
            bboxes = [[bboxes[1:] for bboxes in label_] for label_ in label]
            cls_ids = [[bboxes[0] for bboxes in label_] for label_ in label]

            tot_frs = 17 if self.tg_mode else 16

            in_data = {'image': data[0]}
            [in_data.update({f'image{i}': data[i]}) for i in range(1, tot_frs)]
            
            in_data.update({'bboxes': bboxes[0]})
            [in_data.update({f'bboxes{i}': bboxes[i]}) for i in range(1, tot_frs)]

            in_data.update({'category_ids': cls_ids[0]})
            [in_data.update({f'category_ids{i}': cls_ids[i]}) for i in range(1, tot_frs)]

            if self.tg_mode:
                #-----------------------------------------------------------------------#
                #                       Processing the RGB data                         #
                #-----------------------------------------------------------------------#
                tr_data = self.transform[0](**in_data)
                tr_imgs = torch.stack([torch.from_numpy(tr_data[key]) for key in list(in_data.keys())[:tot_frs]]).permute(3, 0, 1, 2)
                rgb_data = tr_imgs[:, :16, :, :]
                
                last_fr_bboxes = tr_data['bboxes15']
                last_fr_ids = tr_data['category_ids15']                
                annot_list = [[last_fr_ids[i]] + [*last_fr_bboxes[i]] for i in range(len(last_fr_ids))]
                labels = self.prep_targets.build_targets(annot_list)
                
                #-----------------------------------------------------------------------#
                #                       Processing the TG data                          #
                #-----------------------------------------------------------------------#
                
                tg_data = self.transform[1](**in_data)
                tg_imgs = torch.stack([torch.from_numpy(tg_data[key]) for key in list(in_data.keys())[:tot_frs]]).permute(3, 0, 1, 2)
                tg_imgs = (1 + torch.diff(tg_imgs, dim=1)) / 2
                
                last_tg_fr_bboxes = tg_data['bboxes15']
                last_tg_fr_ids = tg_data['category_ids15']                
                tg_annot_list = [[last_tg_fr_ids[i]] + [*last_tg_fr_bboxes[i]] for i in range(len(last_tg_fr_ids))]
                tg_labels = self.prep_targets.build_targets(tg_annot_list)
                
                return (rgb_data, tg_imgs), labels, tg_labels
            
            else:
                tr_data = self.transform(**in_data)

                last_fr_bboxes = tr_data['bboxes15']
                last_fr_ids = tr_data['category_ids15']
                
                annot_list = [[last_fr_ids[i]] + [*last_fr_bboxes[i]] for i in range(len(last_fr_ids))]
                labels = self.prep_targets.build_targets(annot_list)

                rgb_data = torch.stack([torch.from_numpy(tr_data[key]) for key in list(in_data.keys())[:tot_frs]]).permute(3, 0, 1, 2)
                
                return rgb_data, labels
        
        elif self.mode == 'val':
            data = self.load_clip(self.dataset_df['data'][idx])
            label = self.dataset_df['built_GTs'][idx]

            if self.cfg.arch_cfg.tg_cfg.type == 'concat':       #! Deprecated!
                rgb_clip = self.transform[0](data)     # shape: (C, n_fr, H, W)
                tg_clip = (1 + torch.diff(rgb_clip, dim=1)) / 2
                rgb_clip = rgb_clip[:, 1:, :, :]        # The first one has 17 frames to use for TG calculation

                return self.transform[1](rgb_clip), label, self.transform[1](tg_clip)

            return self.transform(data), label

        else:
            label = self.dataset_df['built_GTs'][idx]       # Already a list
            frame_names = self.dataset_df['frame_names'][idx]

            if self.cfg.arch_cfg.tg_cfg.type == 'concat':       #! Deprecated!
                # List of all clips in the video
                rgb_data = [self.transform[0](self.load_clip(data_)) for data_ in self.dataset_df['data'][idx]]
                tg_data = [self.transform[1]((1 + torch.diff(rgb_clip, dim=1)) / 2) for rgb_clip in rgb_data]

                # The first one has 17 frames to use for TG calculation
                rgb_data = [self.transform[1](rgb_clip[:, 1:, :, :]) for rgb_clip in rgb_data]
                
                return rgb_data, label, frame_names, tg_data
            else:
                data = [self.transform(self.load_clip(data_)) for data_ in self.dataset_df['data'][idx]]    # List of all clips in the video

            return data, label, frame_names

    def load_clip(self, clip):
        """
            This method reads the clip frames from the input path_list and returns the result in a list.
        :param clip:
        :return: list of clip frames
        """
        if self.cfg.train_cfg.load_ds_to_mem:
            # In this case the clip has the indices in the h5py dset
            loaded_clip = [self.loaded_ds[idx] for idx in clip]
        else:
            loaded_clip = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in clip]
        return loaded_clip

    @staticmethod
    def visualize_bbox(img, bbox, class_name, thickness=2):

        BOX_COLOR = (255, 0, 0) # Red

        """Visualizes a single bounding box on the image"""
        cx, cy, w, h = bbox
        cx *= img.shape[1]
        w *= img.shape[1]
        cy *= img.shape[0]
        h *= img.shape[0]
        
        x_min, x_max, y_min, y_max = cx - w/2, cx + w/2, cy - h/2, cy + h/2
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=thickness)
        
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)    
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.65, 
            color=(255, 255, 255), # White 
            lineType=cv2.LINE_AA,
        )
        return img

    def visualize(self, image, bboxes, category_ids, category_id_to_name):
        img = image.copy()
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = self.visualize_bbox(img, bbox, class_name)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)