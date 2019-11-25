#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os
from data.config import cfg
import cv2
import numpy as np
import random

WIDER_ROOT = os.path.join(cfg.HOME, 'WIDER')
train_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                               'wider_face_train_bbx_gt.txt')
val_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                             'wider_face_val_bbx_gt.txt')

WIDER_TRAIN = os.path.join(WIDER_ROOT, 'WIDER_train', 'images')
WIDER_VAL = os.path.join(WIDER_ROOT, 'WIDER_val', 'images')


def parse_wider_file(root, file):
    with open(file, 'r') as fr:
        lines = fr.readlines()
    face_count = []
    img_paths = []
    face_loc = []
    img_faces = []
    count = 0
    flag = False
    for k, line in enumerate(lines):
        line = line.strip().strip('\n')
        if count > 0:
            line = line.split(' ')
            count -= 1
            loc = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
            face_loc += [loc]
        if flag:
            face_count += [int(line)]
            flag = False
            count = int(line)
        if 'jpg' in line:
            img_paths += [os.path.join(root, line)]
            flag = True

    total_face = 0
    for k in face_count:
        face_ = []
        for x in xrange(total_face, total_face + k):
            face_.append(face_loc[x])
        img_faces += [face_]
        total_face += k
    return img_paths, img_faces


def wider_data_file():
    img_paths, bbox = parse_wider_file(WIDER_TRAIN, train_list_file)
    fw = open(cfg.FACE.TRAIN_FILE, 'w')
    for index in xrange(len(img_paths)):
        path = img_paths[index]
        boxes = bbox[index]
        fw.write(path)
        fw.write(' {}'.format(len(boxes)))
        for box in boxes:
            data = ' {} {} {} {} {}'.format(box[0], box[1], box[2], box[3], 1)
            fw.write(data)
        fw.write('\n')
    fw.close()

    img_paths, bbox = parse_wider_file(WIDER_VAL, val_list_file)
    fw = open(cfg.FACE.VAL_FILE, 'w')
    for index in xrange(len(img_paths)):
        path = img_paths[index]
        boxes = bbox[index]
        fw.write(path)
        fw.write(' {}'.format(len(boxes)))
        for box in boxes:
            data = ' {} {} {} {} {}'.format(box[0], box[1], box[2], box[3], 1)
            fw.write(data)
        fw.write('\n')
    fw.close()

def pad_to_square(image):
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :,:] = 0
    image_t[0:0 + height, 0:0 + width] = image
    return image_t
def rescale_gt(img,gt):
    img_h, img_w = img.shape[:2]
    scale_w,scale_h = float(640)/img_w,float(640)/img_h
    gt[:,0::2] *= scale_w
    gt[:,1::2] *= scale_h
    img = cv2.resize(img,(640,640))
    # ratio = max(img_h, img_w) / float(self.img_size)
    # new_h = int(img_h / ratio)
    # new_w = int(img_w / ratio)
    # ox = (self.img_size - new_w) // 2
    # oy = (self.img_size - new_h) // 2
    # scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # out = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8) 
    # out[oy:oy + new_h, ox:ox + new_w, :] = scaled
    # window = [ox,oy,new_w,new_h]
    # return out, window
    return img,gt

def showimg(imgpath,imgdir):
    fp = open(imgpath,'r')
    cnts = fp.readlines()
    idx = list(range(len(cnts)))
    random.shuffle(idx)
    for tmp in cnts:
        #tmp = cnts[i]
        tmp_spl = tmp.strip().split(',')
        if tmp_spl[0] != '283554/f8e0000ae0930df.jpg': #'283081/1bb1e000adc2670f.jpg': ##'284193/20c22000f698c02e.jpg': #
            continue
        tmppath = os.path.join(imgdir,tmp_spl[0])
        img = cv2.imread(tmppath)
        dets = list(map(float,tmp_spl[1:]))
        dets = np.reshape(dets,[-1,5])
        dets = dets[:,:4]
        h,w = img.shape[:2]
        print('shape:',h,w)
        a = max(h,w)
        det_bef = dets
        det_test = dets.copy()
        det_test[:,0::2]  /= float(w)
        det_test[:,1::2]  /= float(h)
        det_test*=700
        #print('test:',det_test)
        #det_bef = det_bef.astype(int)
        # print('before:',det_bef)
        dets[:,0] = w-det_bef[:,2]-1
        dets[:,2] = w-det_bef[:,0]-1
        # dets = dets.astype(int)
        # print('after:',dets)
        #img = pad_to_square(img)
        dets[:,0::2]  /= float(w)
        dets[:,1::2]  /= float(h)
        img = cv2.resize(img,(700,700))
        cv2.imshow('org',img)
        a = img.shape[0]
        img = img[:,::-1,:]
        img = cv2.resize(img,(700,700))
        # dets *=700
        print('after',dets)
        for i in range(dets.shape[0]):
            x1,y1,x2,y2 = dets[i,:4] *700
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.imshow('src',img)
        cv2.waitKey(0)
        print(tmppath)
        
if __name__ == '__main__':
    #wider_data_file()
    fpath = 'crowedhuman_train.txt'
    imgdir = '/data/detect/HollywoodHeads/JPEGImages'
    imgdir = '/data/detect/head/imgs'
    showimg(fpath,imgdir)