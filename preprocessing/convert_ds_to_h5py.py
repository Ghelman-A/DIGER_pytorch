"""
    This script converts the Excavator dataset to h5py format for faster access during
    model training and inference.

    There are two outputs in this script:
        - The entire dataset loaded in cv2 format and converted to the given size in h5py format
        - A dict file that has the idx, label, and the frame number for the data in the h5py file
    Author: Ali Ghelmani,       Date: Dec. 06, 2022
"""
import h5py
from pathlib import Path
import numpy as np
import cv2
from cfg.config import diger_cfg
import pickle

ds_dir = Path(diger_cfg.raw_vid_dir)

index_dict = dict()
new_size = (int(diger_cfg.data_prep.fr_width / diger_cfg.data_prep.resize_factor), 
            int(diger_cfg.data_prep.fr_height / diger_cfg.data_prep.resize_factor))

#---------------------------------------------------------------------------------#
# Counting the frames at the start to allocate the required memory size, otherwise#
# it's too slow!                                                                  #
#---------------------------------------------------------------------------------#
total_fr_in_dset = 0
for folder in ds_dir.iterdir():
    total_fr_in_dset += len(list(folder.glob('*.PNG')))
ds_array = np.empty(shape=(total_fr_in_dset, new_size[1], new_size[0], 3), dtype=np.uint8)   # The last 3 is for RGB channels


#---------------------------------------------------------------------------------#
# Creating the h5py dset file                                                     #
#---------------------------------------------------------------------------------#

img_count = 0
for idx, folder in enumerate(sorted(ds_dir.iterdir())):
    print(idx, folder.name)

    for frame in sorted(list(folder.glob("*.PNG"))):
        index_dict[str(frame)] = img_count
        img = cv2.cvtColor(cv2.imread(str(frame)), cv2.COLOR_BGR2RGB)
        ds_array[img_count] = cv2.resize(img, dsize=new_size, interpolation=cv2.INTER_AREA)

        img_count += 1

print(f"ds_array shape: {ds_array.shape}")

# Writing the created list to a h5py file
f = h5py.File(f'{diger_cfg.ds_list_dir}/Excavator_ds.hdf5', 'w', libver='latest')
dset = f.create_dataset('dset', data=ds_array)
f.close()

with open(f"{diger_cfg.ds_list_dir}/Excavator_dset_idx.pkl", 'wb') as file:
    pickle.dump(index_dict, file)
