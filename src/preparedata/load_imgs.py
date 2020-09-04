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
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg


class ReadDataset(u_data.Dataset): #data.Dataset
    """
    VOC Detection Dataset Object
    """
    def __init__(self,imgfiles,imgdir,train_mode='train'):
        self.crowhuman_file = imgfiles
        self.img_w = cfg.resize_width
        self.img_h = cfg.resize_height
        self.crowhuman_dir = imgdir
        self.ids = list()
        self.annotations = []
        self.load_txt()
        self.total_num = self.__len__()
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        self.rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        self.training = train_mode

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
            img_name = tmp_splits[0].split('/')[-1][:-4] if len(tmp_splits[0].split('/')) >0 else tmp_splits[0][:-4]
            self.ids.append((self.crowhuman_dir,img_name))
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
        # img_pro,gt_pro = self.prepro(img_data,gt_box_label)
        if self.training=='train':
            img_data,gt_box_label = self.mirror(img_data,gt_box_label)
        img_pro,gt_pro = self.resize_subtract_mean(img_data,gt_box_label)
        return torch.from_numpy(img_pro).permute(2, 0, 1),gt_pro

    def pull_image(self,index):
        tmp_annotation = self.annotations[index]
        tmp_path = tmp_annotation[0]
        img_data = cv2.imread(tmp_path)
        img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
        return img_data

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
        
    def norm_gt(self,image,boxes_f):
        height, width, _ = image.shape
        boxes_f[:,0] = boxes_f[:,0] / float(width)
        boxes_f[:,2] = boxes_f[:,2] / float(width)
        boxes_f[:,1] = boxes_f[:,1] / float(height)
        boxes_f[:,3] = boxes_f[:,3] / float(height)
        return boxes_f

    def resize_subtract_mean(self,image,gt):
        # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        # interp_method = interp_methods[random.randrange(5)]
        height,width = image.shape[:2]
        if height < self.img_h or width < self.img_w:
            image,gt = self.rescale(image,gt,height,width)
        image,gt = self.cropimg(image,gt)
        gt = self.norm_gt(image,gt)
        # image = cv2.resize(image, (insize, insize), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32)
        # image = image / 255.0
        image -= self.rgb_mean
        return image,gt

    def rescale(self,image,boxes_f,height,width):
        # min_side = min(height,width)
        scale = (self.img_w+10.0)/width if (self.img_w+10.0)/width > (self.img_h+10.0)/height else (self.img_h+10.0)/height
        n_w = width * scale
        n_h = height * scale
        boxes_f[:,0] = boxes_f[:,0] / float(width) * n_w
        boxes_f[:,2] = boxes_f[:,2] / float(width) * n_w
        boxes_f[:,1] = boxes_f[:,1] / float(height) * n_h
        boxes_f[:,3] = boxes_f[:,3] / float(height) * n_h
        image = cv2.resize(image,(int(n_w),int(n_h)))
        return image,boxes_f

    def cropimg(self,image,gt_box):
        h,w = image.shape[:2]
        while 1:
            dh,dw = int(random.random()*(h-self.img_h)),int(random.random()*(w-self.img_w))
            nx1 = dw
            nx2 = dw+self.img_w
            ny1 = dh
            ny2 = dh+self.img_h
            img = image[ny1:ny2,nx1:nx2,:]
            # gt = gt[dh:(dh+cfgs.IMGHeight),dw:(dw+cfgs.IMGWidth)]
            gt = gt_box.copy()
            keep_idx = np.where(gt[:,2]>nx1)
            gt = gt[keep_idx]
            keep_idx = np.where(gt[:,0]<nx2)
            gt = gt[keep_idx]
            keep_idx = np.where(gt[:,3]>ny1)
            gt = gt[keep_idx]
            keep_idx = np.where(gt[:,1]<ny2)
            gt = gt[keep_idx]
            gt[:,0] = np.clip(gt[:,0],nx1,nx2)-nx1
            gt[:,2] = np.clip(gt[:,2],nx1,nx2)-nx1
            gt[:,1] = np.clip(gt[:,1],ny1,ny2)-ny1
            gt[:,3] = np.clip(gt[:,3],ny1,ny2)-ny1
            if len(gt)>0:
                break
        return img,gt

    def crop(self,image, boxes, labels, img_dim):
        height, width, _ = image.shape
        short_side = min(width, height)
        if short_side > 3*img_dim:
            PRE_SCALES = [0.2,0.3]
        elif short_side > 2*img_dim:
            PRE_SCALES = [0.3,0.4]
        elif short_side > img_dim:
            PRE_SCALES = [0.4,0.5]
        else:
            PRE_SCALES = [0.6,0.8]
        if random.randrange(2):
            for _ in range(20):
                scale = random.choice(PRE_SCALES)
                if short_side < 450:
                    scale = 1.0
                w = int(scale * short_side)
                h = w
                if width == w:
                    l = 0
                else:
                    l = random.randrange(width - w)
                if height == h:
                    t = 0
                else:
                    t = random.randrange(height - h)
                roi = np.array((l, t, l + w, t + h))
                value = self.matrix_iof(boxes, roi[np.newaxis])
                flag = (value >= 1)
                if not flag.any():
                    continue
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
                boxes_t = boxes[mask_a].copy()
                labels_t = labels[mask_a].copy()
                if boxes_t.shape[0] == 0:
                    continue
                image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
                boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
                boxes_t[:, :2] -= roi[:2]
                boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
                boxes_t[:, 2:] -= roi[:2]
                # make sure that the cropped image contains at least one face > 16 pixel at training image scale
                b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
                b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
                mask_b = np.minimum(b_w_t, b_h_t) > 0.0
                boxes_t = boxes_t[mask_b]
                labels_t = labels_t[mask_b]
                if boxes_t.shape[0] == 0:
                    continue
                labels_t = np.expand_dims(labels_t, 1)
                targets_t = np.hstack((boxes_t, labels_t))
                return image_t, targets_t
        labels = np.expand_dims(labels, 1)
        targets = np.hstack((boxes, labels))
        return image, targets
    def matrix_iof(self,a, b):
        """
        return iof of a and b, numpy version for data augenmentation
        """
        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / np.maximum(area_a[:, np.newaxis], 1)



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