import numpy as np
import os

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(float(box1[0]-box1[2]/2.0), float(box2[0]-box2[2]/2.0))
        Mx = max(float(box1[0]+box1[2]/2.0), float(box2[0]+box2[2]/2.0))
        my = min(float(box1[1]-box1[3]/2.0), float(box2[1]-box2[3]/2.0))
        My = max(float(box1[1]+box1[3]/2.0), float(box2[1]+box2[3]/2.0))
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def compute_score_one_class(bbox1, bbox2, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbx: <x1> <y1> <x2> <y2> <class score>
    n_bbox1 = bbox1.shape[0]
    n_bbox2 = bbox2.shape[0]
    
    # for saving all possible scores between each two bbxes in successive frames
    scores = np.zeros([n_bbox1, n_bbox2], dtype=np.float32)
    for i in range(n_bbox1):
    
        box1 = bbox1[i, :4]
        
        for j in range(n_bbox2):
            box2 = bbox2[j, :4]
            
            bbox_iou_frames = bbox_iou(box1, box2, x1y1x2y2=True)
           
            sum_score_frames = bbox1[i, 4] + bbox2[j, 4]
            mul_score_frames = bbox1[i, 4] * bbox2[j, 4]
            
            scores[i, j] = w_iou * bbox_iou_frames + w_scores * sum_score_frames + w_scores_mul * mul_score_frames

    return scores

def link_bbxes_between_frames(bbox_list, w_iou=1.3, w_scores=1.0, w_scores_mul=0.5):
    # bbx_list: list of bounding boxes <x1> <y1> <x2> <y2> <class score>
    # check no empty detections
    ind_notempty = []
    nfr = len(bbox_list)
    for i in range(nfr):
        if np.array(bbox_list[i]).size:
            ind_notempty.append(i)
    
    # no detections at all
    if not ind_notempty:
        return []
    
    # miss some frames
    elif len(ind_notempty)!=nfr:
        for i in range(nfr):
            if not np.array(bbox_list[i]).size:
                
                # copy the nearest detections to fill in the missing frames
                ind_dis = np.abs(np.array(ind_notempty) - i)
                nn = np.argmin(ind_dis)
                bbox_list[i] = bbox_list[ind_notempty[nn]]

    detect = bbox_list
    nframes = len(detect)
    res = []
    isempty_vertex = np.zeros([nframes,], dtype=np.bool)
    edge_scores = [compute_score_one_class(detect[i], detect[i+1], w_iou=w_iou, 
                                           w_scores=w_scores, w_scores_mul=w_scores_mul) for i in range(nframes-1)]
    copy_edge_scores = edge_scores

    while not np.any(isempty_vertex):
        # initialize
        scores = [np.zeros([d.shape[0],], dtype=np.float32) for d in detect]
        index = [np.nan*np.ones([d.shape[0],], dtype=np.float32) for d in detect]
        
        # viterbi
        # from the second last frame back
        for i in range(nframes-2, -1, -1):
            edge_score = edge_scores[i] + scores[i+1]
            # find the maximum score for each bbox in the i-th frame and the corresponding index
            scores[i] = np.max(edge_score, axis=1)
            index[i] = np.argmax(edge_score, axis=1)
        
        # decode
        idx = -np.ones([nframes], dtype=np.int32)
        idx[0] = np.argmax(scores[0])
        for i in range(0, nframes-1):
            idx[i+1] = index[i][idx[i]]
        
        # remove covered boxes and build output structures
        this = np.empty((nframes, 6), dtype=np.float32)
        this[:, 0] = 1 + np.arange(nframes)
        for i in range(nframes):
            j = idx[i]
            iouscore = 0
            if i < nframes-1:
                iouscore = copy_edge_scores[i][j, idx[i+1]] - bbox_list[i][j, 4] - bbox_list[i+1][idx[i+1], 4]

            if i < nframes-1: edge_scores[i] = np.delete(edge_scores[i], j, 0)
            if i > 0: edge_scores[i-1] = np.delete(edge_scores[i-1], j, 1)
            this[i, 1:5] = detect[i][j, :4]
            this[i, 5] = detect[i][j, 4]
            detect[i] = np.delete(detect[i], j, 0)
            isempty_vertex[i] = (detect[i].size==0) # it is true when there is no detection in any frame
        res.append( this )
        if len(res) == 3:
            break

    return res

def overlap2d(b1, b2):
    xmin = np.maximum( b1[:,0], b2[:,0] )
    xmax = np.minimum( b1[:,2]+1, b2[:,2]+1)
    width = np.maximum(0, xmax-xmin)
    ymin = np.maximum( b1[:,1], b2[:,1] )
    ymax = np.minimum( b1[:,3]+1, b2[:,3]+1)
    height = np.maximum(0, ymax-ymin)   
    return width*height

def area2d(b):
    return (b[:,2]-b[:,0]+1)*(b[:,3]-b[:,1]+1)

def iou3d(b1, b2):
    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:,0] == b2[:,0])
    o = overlap2d(b1[:,1:5],b2[:,1:5])
    return np.mean( o/(area2d(b1[:,1:5])+area2d(b2[:,1:5])-o) )

def iou3dt(b1, b2):
    tmin = max(b1[0,0], b2[0,0])
    tmax = min(b1[-1,0], b2[-1,0])
    if tmax <= tmin: return 0.0    
    temporal_inter = tmax-tmin+1
    temporal_union = max(b1[-1,0], b2[-1,0]) - min(b1[0,0], b2[0,0]) + 1 
    return iou3d(b1[np.where(b1[:,0]==tmin)[0][0]:np.where(b1[:,0]==tmax)[0][0]+1,:] , 
                 b2[np.where(b2[:,0]==tmin)[0][0]:np.where(b2[:,0]==tmax)[0][0]+1,:]  ) * temporal_inter / temporal_union

def nms_3d(detections, overlap=0.5):
    # detections: [(tube1, score1), (tube2, score2)]
    if len(detections) == 0:
        return np.array([], dtype=np.int32)
    I = np.argsort([d[1] for d in detections])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0
    while I.size>0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([ iou3dt(detections[ii][0],detections[i][0]) for ii in I[:-1] ])
        I  = I[np.where(ious<=overlap)[0]]
    return indices[:counter]

def link_video_one_class(vid_det, bNMS3d = False):
    '''
    linking for one class in a video (in full length)
    vid_det: a list of [frame_index, [bbox cls_score]]
    gtlen: the mean length of gt in training set
    return a list of tube [array[frame_index, x1,y1,x2,y2, cls_score]]
    '''
    # list of bbox information [[bbox in frame 1], [bbox in frame 2], ...]
    vdets = [vid_det[i][1] for i in range(len(vid_det))]
    vres = link_bbxes_between_frames(vdets)
    
    if len(vres) != 0:
        if bNMS3d:
            tube = [b[:, :5] for b in vres]
            # compute score for each tube
            tube_scores = [np.mean(b[:, 5]) for b in vres]
            dets = [(tube[t], tube_scores[t]) for t in range(len(tube))]
            
            # nms for tubes
            keep = nms_3d(dets, 0.3) # bug for nms3dt
            if np.array(keep).size:
                vres_keep = [vres[k] for k in keep]
                
                # max subarray with penalization -|Lc-L|/Lc
                vres = vres_keep
    return vres

def video_ap_one_class(pred_videos):
    '''
        pred_videos: [ video_index, [ [frame_index, [[x1,y1,x2,y2, score]] ] ] ]
    '''
    # link for prediction
    pred = []

    for pred_v in pred_videos:
        video_index = pred_v[0]
        pred_link_v = link_video_one_class(pred_v[1], True) # [array<frame_index, x1,y1,x2,y2, cls_score>]
        
        for tube in pred_link_v:
            pred.append((video_index, tube))

    return pred

def evaluate_videoAP(all_boxes, CLASSES):
    '''
        all_boxes: {imgname:{cls_ind:array[x1,y1,x2,y2, cls_score]}}
    '''
    def imagebox_to_videts(img_boxes, CLASSES):
        # image names
        keys = list(all_boxes.keys())
        keys.sort()
        res = []
        # without 'background'
        for cls_ind, cls in enumerate(CLASSES[0:]):
            v_cnt = 1
            frame_index = 1
            v_dets = []
            cls_ind += 1
            # get the directory path of images
            preVideo = keys[0][0:-7]
            for i in range(len(keys)):
                curVideo = keys[i][0:-7]
                img_cls_dets = img_boxes[keys[i]][cls_ind]
                v_dets.append([frame_index, img_cls_dets])
                frame_index += 1
                if preVideo!=curVideo:
                    preVideo = curVideo
                    frame_index = 1
                    del v_dets[-1]
                    res.append([cls_ind, v_cnt, v_dets])
                    v_cnt += 1
                    v_dets = []
                    v_dets.append([frame_index, img_cls_dets])
                    frame_index += 1
            # the last video
            res.append([cls_ind, v_cnt, v_dets])
        return res
    
    pred_videos_format = imagebox_to_videts(all_boxes, CLASSES)
    ap_all = []
    
    for cls_ind, cls in enumerate(CLASSES[0:]):
        
        cls_ind += 1
        
        pred_cls = [p[1:] for p in pred_videos_format if p[0]==cls_ind]
        
        ap = video_ap_one_class(pred_cls)
        ap_all.append(ap)

    return ap_all

