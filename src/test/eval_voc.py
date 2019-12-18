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
"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""
import sys
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import math
import pickle
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
from s3fd_test import build_s3fd
from detection import Detect
sys.path.append(os.path.join(os.path.dirname(__file__),'../preparedata'))
from vochead import VOCDetection, VOCAnnotationTransform
from load_imgs import ReadDataset


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root',type=str, default=None,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--dataname', default='scut',
                    type=str, help='val dataset name')
parser.add_argument('--dataset', default='Part_A',
                    type=str, help='test dataset')
parser.add_argument('--val_file', default='/wdc/LXY.data/',
                    type=str, help='test file')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset = 'SCUT_HEAD_'+args.dataset
annopath = os.path.join(args.voc_root, dataset, 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, dataset, 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, dataset, 'ImageSets',
                          'Main', 'test.txt')
labelmap = ['person']
#annopath = args.val_file

def test_net(save_folder, net,detector, dataset, top_k,thresh=0.05):
    '''
    get all detections: result shape is [batch,class_num,topk,5]
    save all detections into all_boxes:the shape [cls,image_num,N,5], N is diffient,
                5 clums is (x1, y1, x2, y2, score)
    '''
    num_images = dataset.__len__()
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]
    rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
    print('test all box:',np.shape(all_boxes))
    # build save dir
    output_dir = save_folder
    det_file = os.path.join(output_dir, 'detections.pkl')
    if os.path.exists(det_file):
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
    else:
        for i in tqdm(range(num_images)):
            img = dataset.pull_image(i)
            h, w, _ = img.shape
            scale = torch.Tensor([w, h,w, h])
            image = cv2.resize(img,(640,640))
            image = image.astype(np.float32)
            # image = image / 255.0
            image -= rgb_mean
            image = np.transpose(image,(2,0,1))
            x = Variable(torch.from_numpy(image).unsqueeze(0),requires_grad=False)
            if use_cuda:
                x = x.cuda()
            output = net(x)
            detections = detector(output[0],output[1],output[2]).data
            for j in range(1,detections.size(1)):
                dets = detections[0, j, :]
                mask =  dets[:, 0].gt(thresh).unsqueeze(1).expand_as(dets)
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:] * scale
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                    scores[:, np.newaxis])).astype(np.float32,
                                                                    copy=False)
                all_boxes[j][i] = cls_dets
                #save resuts
            #     fin_mask = np.where(scores > 0.6)[0]
            #     bboxes = boxes.cpu().numpy()[fin_mask]
            #     scores = scores[fin_mask]
            #     for k in range(len(scores)):
            #         leftup = (int(bboxes[k][0]), int(bboxes[k][1]))
            #         right_bottom = (int(bboxes[k][2]), int(bboxes[k][3]))
            #         cv2.rectangle(img, leftup, right_bottom, (0, 255, 0), 2)
            # save_file = os.path.join(output_dir, '{}.jpg'.format(i + 1))
            # cv2.imwrite(save_file,img)
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        write_voc_results_file(all_boxes, dataset,output_dir)
    print('Evaluating detections')
    #f = open(det_file,'rb')
    #all_boxes = pickle.load(f)
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset,output_dir)
    do_python_eval(output_dir)

def write_voc_results_file(all_boxes, dataset,save_dir):
    '''
    save results as the follow format: every line includes one record bbox
    imagename score x1 y1 x2 y2
    '''
    for cls_ind, cls_name in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls_name))
        filename = get_voc_results_file_template(cls_name,save_dir)
        tmp_f = open(filename, 'wt')
        for im_ind, index in enumerate(dataset.ids):
            dets = all_boxes[cls_ind+1][im_ind]
            if dets == []:
                continue
            # the VOCdevkit expects 1-based indices
            for k in range(dets.shape[0]):
                tmp_f.write('{:s},{:.3f},{:.1f},{:.1f},{:.1f},{:.1f}\n'.
                    format(index[1],dets[k,-1],dets[k,0],dets[k,1],dets[k,2],dets[k,3]))
        tmp_f.close()

def get_voc_results_file_template(cls_name,save_dir):
    # VOCdevkit/VOC2007/detect_out/det_aeroplane.txt
    filename = 'det' + '_%s.txt' % (cls_name)
    filedir = os.path.join(save_dir, 'detect_out')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

def do_python_eval(output_dir='output', use_07=True):
    '''
    get mAP use07, mean precision when roc in [0,1.1,0.1]
    '''
    cachedir = os.path.join(output_dir, 'gt_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    for i, cls_name in enumerate(labelmap):
        filename = get_voc_results_file_template(cls_name,output_dir)
        rec, prec, ap, pos_num = voc_eval(filename, annopath, imgsetpath, cls_name, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls_name, ap))
        with open(os.path.join(output_dir, cls_name + '_pr.txt'), 'w') as f:
            #pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            print(len(rec),len(prec))
            for tmp_id in range(len(rec)):
                f.write("rec: {:.3f},prec: {:.3f}\n".format(rec[tmp_id],prec[tmp_id]))
            f.write('ap: {:.3f}, pos_num: {}'.format(ap,pos_num))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')

def voc_eval(detfile,annopath,imagesetfile,classname,cachedir,ovthresh=0.5,use_07_metric=True):
    '''Top level function that does the PASCAL VOC evaluation.
    detpath: detection results save path
    annopath: Path to ground_truth annotations, xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for saving the gt annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
    '''
    #get gt
    #class_recs,npos = get_annotions(cachedir,imagesetfile,annopath,classname)
    class_recs,npos = load_annotations(annopath,classname)
    # read dets
    boxes,image_ids = get_detect_results(detfile)
    if len(boxes) >1:
        # go down dets and mark TPs and FPs
        detect_nums = len(image_ids)
        tp = np.zeros(detect_nums)
        fp = np.zeros(detect_nums)
        for idx in range(detect_nums):
            annotations_gt = class_recs[image_ids[idx]]
            bb = boxes[idx, :].astype(float)
            ovmax = -np.inf
            bbox_gt = annotations_gt['bbox'].astype(float)
            ovmax,jmax = get_IoU(bbox_gt,bb)
            #calculate the tp and fp, fp: not match and wrong match
            if ovmax > ovthresh:
                if not annotations_gt['difficult'][jmax]:
                    if not annotations_gt['det'][jmax]:
                        tp[idx] = 1.
                        annotations_gt['det'][jmax] = 1
                    else:
                        fp[idx] = 1.
            else:
                fp[idx] = 1.
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
    return rec, prec, ap,npos

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

def get_annotions(cachedir,imagesetfile,annopath,classname):
    '''
    load and save annotions from annopath
    '''
    # first load gt
    if not os.path.exists(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    imgset_f = open(imagesetfile, 'r')
    lines = imgset_f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # save annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(1 - difficult)
        class_recs[imagename] = {'bbox': bbox,'difficult': difficult,'det': det}
    imgset_f.close()
    return class_recs,npos

def load_annotations(imagesetfile,classname):
    # read list of images
    imgset_f = open(imagesetfile, 'r')
    lines = imgset_f.readlines()
    class_recs = {}
    npos = 0
    label_num = labelmap.index(classname)+1
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

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)
    return objects

def get_detect_results(detfile):
    '''
    load detects from detfile
    '''
    with open(detfile, 'r') as f:
        detection_cnts = f.readlines()
    image_ids = []
    confidence = []
    boxes = []
    if len(detection_cnts) >=1:
        for tmp_cnt in detection_cnts:
            tmp_splits = tmp_cnt.strip().split(',')
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
    return boxes,image_ids

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        rec_num = 0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
                rec_num +=1
            ap = ap + p
        ap = ap/11.0
        #ap = ap / float(rec_num)
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """
    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi
    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)
    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)
    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]
 
    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
 
    return lamr, mr, fppi

if __name__ == '__main__':
    # load net
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    detector = Detect(cfg)
    if use_cuda:
        net.cuda()
        cudnn.benckmark = True
    print('finish loading model')
    if args.dataname == 'scut':
        dataset = VOCDetection(cfg.HEAD.DIR, image_sets=[(args.dataset, 'test')],
                            target_transform=VOCAnnotationTransform(),
                            mode='test',
                            dataset_name='SCUT')
    elif args.dataname == 'crowedhuman':
        dataset = ReadDataset(args.val_file,args.voc_root)
    test_net(args.save_folder, net,detector, dataset,args.top_k, args.threshold)
    