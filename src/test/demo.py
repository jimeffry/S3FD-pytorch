#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
# from s3fd import S3FD
from vgg16 import S3FD
from prior_box import PriorBox
from detection import Detect,Detect_demo,DetectIou
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from bbox_utils import nms,nms_py

def parms():
    parser = argparse.ArgumentParser(description='s3df demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.05, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--ctx', default=True, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()


class HeadDetect(object):
    def __init__(self,args):
        if args.ctx and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self.loadmodel(args.modelpath)
        self.threshold = args.threshold
        self.img_dir = args.img_dir
        
        self.detect = Detect(cfg)
        # self.detect = DetectIou(cfg)
        # self.detect = Detect_demo(cfg)
        self.Prior = PriorBox()
        with torch.no_grad():
            self.priors =  self.Prior()
        self.num_classes = cfg.NUM_CLASSES

    def loadmodel(self,modelpath):
        if self.use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        # self.net = build_s3fd('test', cfg.NUM_CLASSES)
        self.net = S3FD(cfg.NUM_CLASSES,cfg.NumAnchor).to(device)
        # print(self.net)
        self.net.load_state_dict(torch.load(modelpath,map_location=device))
        self.net.eval()
        # torch.save(self.net.state_dict(),'srd_tr.pth')
    def get_hotmaps(self,conf_maps):
        '''
        conf_maps: feature_pyramid maps for classification
        '''
        hotmaps = []
        print('feature maps num:',len(conf_maps))
        for tmp_map in conf_maps:
            batch,h,w,c = tmp_map.size()
            tmp_map = tmp_map.view(batch,h,w,-1,self.num_classes)
            tmp_map = tmp_map[0,:,:,:,1:]
            tmp_map_soft = torch.nn.functional.softmax(tmp_map,dim=3)
            cls_mask = torch.argmax(tmp_map_soft,dim=3,keepdim=True)
            #score,cls_mask = torch.max(tmp_map_soft,dim=4,keepdim=True)
            #cls_mask = cls_mask.unsqueeze(4).expand_as(tmp_map_soft)
            #print(cls_mask.data.size(),tmp_map_soft.data.size())
            tmp_hotmap = tmp_map_soft.gather(3,cls_mask)
            map_mask = torch.argmax(tmp_hotmap,dim=2,keepdim=True)
            tmp_hotmap = tmp_hotmap.gather(2,map_mask)
            tmp_hotmap.squeeze_(3)
            tmp_hotmap.squeeze_(2)
            print('map max:',tmp_hotmap.data.max())
            hotmaps.append(tmp_hotmap.data.numpy())
        return hotmaps
    def display_hotmap(self,hotmaps):
        '''
        hotmaps: a list of hot map ,every shape is [1,h,w]
        '''       
        row_num = 2
        col_num = 3
        fig, axes = plt.subplots(nrows=row_num, ncols=col_num, constrained_layout=True)
        for i in range(row_num):
            for j in range(col_num):
                #ax_name = 'ax_%s' % (str(i*col_num+j))
                #im_name = 'im_%s' % (str(i*col_num+j))
                ax_name = axes[i,j]
                im_name = ax_name.imshow(hotmaps[i*col_num+j])
                ax_name.set_title("feature_%d" %(i*col_num+j+3))
        #**************************************************************
        img = hotmaps[-1]
        min_d = np.min(img)
        max_d = np.max(img)
        tick_d = []
        while min_d < max_d:
            tick_d.append(min_d)
            min_d+=0.01
        cb4 = fig.colorbar(im_name) #ticks=tick_d)
        plt.savefig('hotmap.png')
        plt.show()
    def propress(self,img):
        rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        img = cv2.resize(img,(cfg.resize_width,cfg.resize_height))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img -= rgb_mean
        #img = img[:,:,::-1]
        img = np.transpose(img,(2,0,1))
        return img
    def inference_img(self,imgorg):
        t1 = time.time()
        img = self.propress(imgorg.copy())
        bt_img = Variable(torch.from_numpy(img).unsqueeze(0))
        if self.use_cuda:
            bt_img = bt_img.cuda()
        output = self.net(bt_img)
        t2 = time.time()
        with torch.no_grad():
            bboxes = self.detect(output[0],output[1],self.priors)
        # bboxes = self.nms_filter(bboxes)
        t3 = time.time()
        # print('consuming:',t2-t1,t3-t2)
        showimg = self.label_show(bboxes,imgorg)
        # return showimg,bboxes.data.cpu().numpy()
        return showimg,bboxes

    def nms_filter(self,bboxes):
        scale = np.array([640,640,640,640])[np.newaxis,]
        boxes = bboxes[0][0] * scale
        scores = bboxes[0][1]
        ids, count = nms_py(boxes, scores, 0.2,1000)
        boxes = boxes[ids[:count]]
        scores = scores[ids[:count]]
        return [[boxes,scores]]

    def label_show(self,rectangles,img):
        rectangles = rectangles.data.cpu().numpy()
        # img = cv2.resize(img,(640,640))
        imgh,imgw,_ = img.shape
        scale = np.array([imgw,imgh,imgw,imgh])
        # print(scale)
        bboxes_score = rectangles[0]
        bboxes = bboxes_score[0]
        scores = bboxes_score[1]
        threslist = [0.3,0.2]
        for i in range(1,rectangles.shape[1]):
            j = 0
            while rectangles[0,i,j,0] >= self.threshold:
                score = rectangles[0,i,j,0]
                dets = rectangles[0,i,j,1:] * scale
                x1,y1,x2,y2 = dets
                min_re = min(y2-y1,x2-x1)
                if min_re <32:
                    thres = 0.2
                else:
                    thres = 0.5
                if score >=thres:
                    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                    txt = "{:.3f}".format(score) # cfg.shownames[i] #
                    point = (int(x1),int(y1-5))
                    cv2.putText(img,txt,point,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                j+=1
        '''
        for j in range(bboxes.shape[0]):
            dets = bboxes[j] 
            score = scores[j]
            x1,y1 = dets[:2]
            x2,y2 = dets[2:]
            min_re = min(y2-y1,x2-x1)
            if min_re < 16:
                thresh = 0.5
            else:
                thresh = 0.9
            if score >= thresh:
                cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                txt = "{:.2f}".format(score)
                point = (int(x1),int(y1-5))
                # cv2.putText(img,txt,point,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        '''
        return img
    def detectheads(self,imgpath):
        if os.path.isdir(imgpath):
            cnts = os.listdir(imgpath)
            for tmp in cnts:
                tmppath = os.path.join(imgpath,tmp.strip())
                img = cv2.imread(tmppath)
                if img is None:
                    continue
                showimg,_ = self.inference_img(img)
                cv2.imshow('demo',showimg)
                cv2.waitKey(0)
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            # if not os.path.exists(self.save_dir):
            #     os.makedirs(self.save_dir)
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                if len(tmp_file.split(','))>0:
                    tmp_splits = tmp_file.split(',')
                    tmp_file = tmp_splits[0]
                    gt_box = map(float,tmp_splits[1:])
                    gt_box = np.array(list(gt_box))
                    gt_box = gt_box.reshape([-1,5])
                if not tmp_file.endswith('jpg'):
                    tmp_file = tmp_file +'.jpeg'
                tmp_path = os.path.join(self.img_dir,tmp_file) 
                if not os.path.exists(tmp_path):
                    print("image path not exist:",tmp_path)
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp)
                    continue
                frame,_ = self.inference_img(img)
                # for idx in range(gt_box.shape[0]):
                #     pt = gt_box[idx,:4]
                #     i = int(gt_box[idx,4])
                #     cv2.rectangle(frame,
                #                 (int(pt[0]), int(pt[1])),
                #                 (int(pt[2]), int(pt[3])),
                #                 (0,0,255), 2)
                h,w = frame.shape[:2]
                if min(h,w) > 1920:
                    frame = cv2.resize(frame,(1920,1080))          
                cv2.imshow('result',frame)
                #savepath = os.path.join(self.save_dir,save_name)
                # cv2.imwrite('test.jpg',frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            cap = cv2.VideoCapture(imgpath)
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame,_ = self.inference_img(img)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            imgname = imgpath.split('/')[-1]
            if img is not None:
                # grab next frame
                # update FPS counter
                frame,odm_maps = self.inference_img(img)
                # hotmaps = self.get_hotmaps(odm_maps)
                # self.display_hotmap(hotmaps)
                # keybindings for display
                cv2.imshow('result',frame)
                cv2.imwrite(imgname,frame)
                key = cv2.waitKey(0) 
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    detector = HeadDetect(args)
    imgpath = args.file_in
    detector.detectheads(imgpath)