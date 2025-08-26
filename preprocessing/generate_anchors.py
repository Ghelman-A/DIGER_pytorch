'''
Created on Dec 12, 2022
@author: Ali Ghelmani

Modified to fit the available dataset and also to partially optimize!
'''
import numpy as np
import random
from pathlib import Path
import xml.etree.ElementTree as ET
from cfg.config import diger_cfg


def IOU(x, centroids):
    similarities = []
    w, h = x
    
    for centroid in centroids:
        c_w, c_h = centroid

        cw = min(w, c_w)
        ch = min(h, c_h)
        
        i_area = cw * ch
        u_area = w * h + c_w * c_h - i_area
        
        similarity = i_area / u_area
        
        similarities.append(similarity) # will become (k,) shape
        
    return np.array(similarities) 

def write_anchors_to_file(centroids):
    
    anchors = centroids.copy()

    for i in range(anchors.shape[0]):
        anchors[i][0] /= diger_cfg.data_prep.fr_width
        anchors[i][1] /= diger_cfg.data_prep.fr_height
         

    widths = anchors[:, 0]
    sorted_anchors = anchors[np.argsort(widths)]
    print('Anchors = ', sorted_anchors)

    save_file = diger_cfg.localization_cfg.anchor_save_dir
    np.savetxt(save_file, sorted_anchors, fmt='%0.2f', delimiter=', ')

def kmeans(data, centroids):
    
    num_data = data.shape[0]
    k, dim = centroids.shape
    prev_assignments = np.ones(num_data) * (-1)    
    iter = 0
    old_distance = np.zeros((num_data, k))

    while True:
        distance = [] 
        iter += 1

        #-----------------------------------------------------------------#
        #               Calculate distance to centroids                   #
        #-----------------------------------------------------------------#
        for i in range(num_data):
            d = 1 - IOU(data[i], centroids)
            distance.append(d)
        distance = np.array(distance) # D.shape = (num_data,k)
        
        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_distance - distance))))
        
        #-----------------------------------------------------------------#
        #                   Assign samples to centroids                   #
        #-----------------------------------------------------------------#
        assignments = np.argmin(distance, axis=1)
        
        if (assignments == prev_assignments).all() :
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids)
            return

        #-----------------------------------------------------------------#
        #                   Calculate new centroids                       #
        #-----------------------------------------------------------------#
        centroid_sums = np.zeros((k, dim), float)
        for i in range(num_data):
            centroid_sums[assignments[i]] += data[i]        
        
        for j in range(k):            
            centroids[j] = centroid_sums[j] / (np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_distance = distance.copy()  

def get_annot_wh(file_name):
    """
        This function parses the xml label files and extracts the width and height of the bounding boxes.

    Args:
        file_name (_type_): The path of the xml label file

    Returns:
        list of tuples: List of the extracted bbox width and height (w, h) for all objects in the frame.
    """
    tr = ET.parse(file_name)
    root = tr.getroot()
    
    annot_info = []
    for obj in root.findall('object'):
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        bbox_w = (xmax - xmin)
        bbox_h = (ymax - ymin)

        annot_info.append((bbox_w, bbox_h))
    
    return annot_info

def main(cfg):

    #----------------------------------------------------------#
    #     Getting a list of all of (w, h) bbox annotations     #
    #----------------------------------------------------------#
    dset_path = Path(cfg['raw_vid_dir'])
    annot_list = []

    for folder in dset_path.iterdir():

        xml_list = list(folder.glob('*.xml'))
        for file in xml_list:
            annot_list += get_annot_wh(str(file))       # extending the list

    #----------------------------------------------------------#
    #     Using kmeans to find the optimal set of anchors      #
    #----------------------------------------------------------#
    annot_list = np.array(annot_list)
    num_clusters = cfg['localization_cfg']['num_anchors']

    indices = [random.randrange(annot_list.shape[0]) for i in range(num_clusters)]
    centroids = annot_list[indices]
    kmeans(annot_list, centroids)
    print('centroids.shape', centroids.shape)


if __name__=="__main__":
    main(diger_cfg)
