#**********************************
#author: lxy
#time: 14:30 2019.7.1
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os
import sys
import torch
import cv2
import numpy as np
import random
import torch.utils.data as u_data
#from convert_to_pickle import label_show
sys.path.append(os.path.join(os.path.dirname(__file__),'./'))
from config import cfg


class ReadDataset(u_data.Dataset): #data.Dataset
    """
    VOC Detection Dataset Object
    """
    def __init__(self,imgfiles,imgdir):
        self.crowhuman_file = imgfiles
        self.img_size = cfg.resize_width
        self.crowhuman_dir = imgdir
        #self.ids = []
        self.annotations = []
        self.load_txt()
        self.idx = 0
        self.total_num = self.__len__()
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        self.rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')

    def __getitem__(self, index):
        im, gt = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.annotations)

    def load_txt(self):
        self.data_r = open(self.crowhuman_file,'r')
        voc_annotations = self.data_r.readlines()
        for tmp in voc_annotations:
            tmp_splits = tmp.strip().split(',')
            img_path = os.path.join(self.crowhuman_dir,tmp_splits[0])
            #self.ids.append((self.crowhuman_dir,tmp_splits[0].split('/')[-1][:-4]))
            bbox = map(float, tmp_splits[1:])
            if not isinstance(bbox,list):
                bbox = list(bbox)
            bbox.insert(0,img_path)
            self.annotations.append(bbox)

    def close_txt(self):
        self.data_r.close()

    def pull_item(self, index):
        '''
        output: img - shape(c,h,w)
                gt_boxes+label: box-(x1,y1,x2,y2)
                label: dataset_class_num 
        '''
        tmp_annotation = self.annotations[index]
        tmp_path = tmp_annotation[0]
        img_data = cv2.imread(tmp_path)
        img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
        gt_box_label = np.array(tmp_annotation[1:],dtype=np.float32).reshape(-1,5)
        #print('load',gt_box_label) 
        img_pro,gt_pro = self.prepro(img_data,gt_box_label)
        return torch.from_numpy(img_pro).permute(2, 0, 1),gt_pro
    
    def prepro(self,img,gt):
        img ,gt = self.mirror(img,gt)
        # img_data = self.pad_to_square(img)
        gt = self.norm_gt(img,gt)
        img_data = self.resize_subtract_mean(img,self.img_size)
        return img_data,gt

    def pad_to_square(self,image):
        height, width, _ = image.shape
        long_side = max(width, height)
        image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
        image_t[:, :,:] = self.rgb_mean
        image_t[0:0 + height, 0:0 + width] = image
        return image_t

    def mirror(self,image, boxes):
        height, width, _ = image.shape
        boxes_tmp = boxes.copy()
        if random.randrange(2):
            image = image[:, ::-1,:]
            boxes[:, 0] = width - boxes_tmp[:, 2] -1
            boxes[:,2] = width - boxes_tmp[:,0] -1
        return image,boxes
        
    def norm_gt (self,image,boxes_f):
        height, width, _ = image.shape
        boxes_f[:,0] = boxes_f[:,0] / float(width)
        boxes_f[:,2] = boxes_f[:,2] / float(width)
        boxes_f[:,1] = boxes_f[:,1] / float(height)
        boxes_f[:,3] = boxes_f[:,3] / float(height)
        return boxes_f

    def resize_subtract_mean(self,image,insize):
        # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        # interp_method = interp_methods[random.randrange(5)]
        image = cv2.resize(image, (insize, insize), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32)
        # image = image / 255.0
        image -= self.rgb_mean
        return image

    def re_scale(self,img):
        img_h, img_w = img.shape[:2]
        ratio = max(img_h, img_w) / float(self.img_size)
        new_h = int(img_h / ratio)
        new_w = int(img_w / ratio)
        ox = (self.img_size - new_w) // 2
        oy = (self.img_size - new_h) // 2
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        out = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8) 
        out[oy:oy + new_h, ox:ox + new_w, :] = scaled
        window = [ox,oy,new_w,new_h]
        return out, window
    
    def descale(self,box,window,img_w,img_h):
        ox,oy,new_w,new_h = window
        xmin, ymin, xmax, ymax = box[:,:,:,1],box[:,:,:,2],box[:,:,:,3],box[:,:,:,4]
        box[:,:,:,1] = (xmin - ox) / float(new_w) * img_w
        box[:,:,:,2] = (ymin - oy) / float(new_h) * img_h
        box[:,:,:,3] = (xmax - ox) / float(new_w) * img_w
        box[:,:,:,4] = (ymax - oy) / float(new_h) * img_h
        '''
        box[:,:,:,1] = np.minimum(np.maximum(xmin * img_w,0),img_w)
        box[:,:,:,2] = np.minimum(np.maximum(ymin * img_h,0),img_h)
        box[:,:,:,3] = np.minimum(np.maximum(xmax * img_w,0),img_w)
        box[:,:,:,4] = np.minimum(np.maximum(ymax * img_h,0),img_h)
        '''
        return box


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.from_numpy(sample[1]).float())
    return torch.stack(imgs, 0), targets

if __name__=='__main__':
    test_d = ReadDataset()
    img_dict = dict()
    i=0
    total = 133644
    while 3-i:
        img, gt = test_d.get_batch(2)
        img_dict['img_data'] = img[0].numpy()
        img_dict['gt'] = gt[0]
        label_show(img_dict)
        #print(gt[0][:,-1])
        #sys.stdout.write('\r>> %d /%d' %(i,total))
        #sys.stdout.flush()
        i+=1
    print(i)