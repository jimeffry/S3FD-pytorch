# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/11/19 10:09
#project: head detect
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  histogram
####################################################
import sys
import os
import numpy as np 
from matplotlib import pyplot as plt 
import argparse
from collections import defaultdict

def parms():
    parser = argparse.ArgumentParser(description='refinedet traing log')
    parser.add_argument('--file_in', default=None,
                        type=str, help='log file')
    parser.add_argument('--data_name', default='voc', type=str,
                        help='traing data name')
    parser.add_argument('--loss_name',type=str,default='total',help='loss')
    parser.add_argument('--conf_thresh',type=float,default=0.0001,help='confidence thresh')
    parser.add_argument('--file2_in',type=str,default=None,help='load file')
    parser.add_argument('--cmd_type',type=str,default=None,help='')
    return parser.parse_args()


def load_annotations(imagesetfile):
    # read list of images
    imgset_f = open(imagesetfile, 'r')
    lines = imgset_f.readlines()
    imgset_f.close()
    class_recs = {}
    npos = 0
    label_num = 1
    for tmp in lines:
        tmp_splits = tmp.strip().split(',')
        img_name = tmp_splits[0].split('/')[-1][:-4] if len(tmp_splits[0].split('/')) >0 else tmp_splits[0][:-4]
        #img_path = os.path.join(args.img_dir,tmp_splits[0])
        bbox_label = map(float, tmp_splits[1:])
        if not isinstance(bbox_label,list):
            bbox_label = list(bbox_label)
        bbox_label = np.reshape(bbox_label,(-1,5))
        bbox = bbox_label[:,:4]
        labels = bbox_label[:,4]
        keep = np.where(labels==label_num)
        bbox = bbox[keep]
        num = bbox.shape[0]
        det = [False]*num
        npos = npos + num
        difficult = np.zeros(num).astype(np.bool)
        class_recs[img_name] = {'bbox':bbox,'difficult':difficult,'det':det}
    return class_recs,npos

def get_detect_results(detfile,conf_thresh=0.05):
    '''
    load detects from detfile
    '''
    det_read = open(detfile, 'r')
    detection_cnts = det_read.readlines()
    image_ids = []
    confidence = []
    boxes = []
    if len(detection_cnts) >=1:
        for tmp_cnt in detection_cnts:
            tmp_splits = tmp_cnt.strip().split(',')
            if float(tmp_splits[1]) < conf_thresh:
                continue
            image_ids.append(tmp_splits[0])
            confidence.append(float(tmp_splits[1]))
            tmp_box = map(float,tmp_splits[2:])
            boxes.append(list(tmp_box))
        confidence = np.array(confidence)
        boxes = np.array(boxes)
        # sort by confidence , could select top k detections boxes
        sorted_ind = np.argsort(confidence)[::-1]
        # sorted_scores = np.sort(confidence)[::-1]
        boxes = boxes[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        confidence = confidence[sorted_ind]
    det_read.close()
    return boxes,image_ids,confidence

def get_IoU(bbox_gt,bb):
    '''
    bbox_gt: ground truth, all boxes in one image
    bb: one record from detection
    '''
    ovmax = 0.0
    jmax = np.inf
    if bbox_gt.size > 0:
        # compute overlaps every detect record with gts
        ixmin = np.maximum(bbox_gt[:, 0], bb[0])
        iymin = np.maximum(bbox_gt[:, 1], bb[1])
        ixmax = np.minimum(bbox_gt[:, 2], bb[2])
        iymax = np.minimum(bbox_gt[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                (bbox_gt[:,2] - bbox_gt[:,0]) *
                (bbox_gt[:,3] - bbox_gt[:,1]) - inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
    return ovmax,jmax

def get_positivedata(detfile,annofile,keydata,conf):
    '''Top level function that does the PASCAL VOC evaluation.
    detpath: detection results save path
    annopath: Path to ground_truth annotations, xml annotations file.
    [ovthresh]: Overlap threshold (default = 0.5)
    '''
    ovthresh = 0.2
    score_st = conf
    tp_data_dict = defaultdict(list)
    fp_data_dict = defaultdict(list)
    keynames = np.array(keydata)
    #get gt
    class_recs,npos = load_annotations(annofile)
    # read dets
    boxes,image_ids,scores = get_detect_results(detfile,score_st)
    if len(boxes) >1:
        # go down dets and mark TPs and FPs
        detect_nums = len(image_ids)
        for idx in range(detect_nums):
            annotations_gt = class_recs[image_ids[idx]]
            bb = boxes[idx, :].astype(float)
            confidence = scores[idx]
            ovmax = -np.inf
            bbox_gt = annotations_gt['bbox'].astype(float)
            ovmax,jmax = get_IoU(bbox_gt,bb)
            #calculate the tp and fp, fp: not match and wrong match
            tmp_min = min(bb[2]-bb[0],bb[3]-bb[1])
            if tmp_min <= 16:
                ovthresh = 0.2
            else:
                ovthresh = 0.5
            keep_ids = np.where(keynames >=tmp_min)
            if ovmax > ovthresh:
                if len(keep_ids[0])>0:
                    tp_data_dict[str(keynames[keep_ids[0][0]])].append(confidence)
                else:
                    tp_data_dict[str(keynames[-1])].append(confidence)
            else:
                if len(keep_ids[0])>0:
                    fp_data_dict[str(keynames[keep_ids[0][0]])].append(confidence)
                else:
                    tp_data_dict[str(keynames[-1])].append(confidence)
    return tp_data_dict,fp_data_dict

def plot_data(datadict,name,keys):
    num_bins = 10
    row_num = 2
    col_num = 3
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num,figsize=(12,5))
    for i in range(row_num):
        for j in range(col_num):
            bin_cnt,bin_data,patchs = axes[i,j].hist(datadict[str(keys[i*col_num+j])],num_bins,normed=0,color='blue',cumulative=0) #range=(0.0,max_bin)
            axes[i,j].set_xlabel('score')
            axes[i,j].set_ylabel('num')
            axes[i,j].set_title('%s-%s' % (name, str(keys[i*col_num+j])))
            axes[i,j].grid(True)
    plt.savefig('../logs/%s.png' % name,format='png')
    plt.show()

def plot_size2score(filein1,filein2,name,conf):
    '''
    datadict: input data. keys are bounding-box size, values are scores
    '''
    keys = [16,32,64,128,256,512] #[24,48,96,192,384,640]
    tp_data_dict,fp_data_dict = get_positivedata(filein1,filein2,keys,conf)
    name_p = '%s_tp' % name
    name_f = '%s_fp' % name
    plot_data(tp_data_dict,name_p,keys)
    plot_data(fp_data_dict,name_f,keys)
    
if __name__=='__main__':
    args = parms()
    file_in = args.file_in
    data_name = args.data_name
    conf = args.conf_thresh
    file2_in = args.file2_in
    if args.cmd_type == 'size2score':
        plot_size2score(file_in,file2_in,data_name,conf)
    else:
        print('please input cmd')
