#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import torch
from egohand import HandDetection
from widerface import WIDERDetection
from vochead import VOCDetection, VOCAnnotationTransform
from load_imgs import ReadDataset

sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg


def dataset_factory(dataset):
    if dataset == 'face':
        train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')
        val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
    if dataset == 'hand':
        train_dataset = WIDERDetection(cfg.HAND.TRAIN_FILE, mode='train')
        val_dataset = WIDERDetection(cfg.HAND.VAL_FILE, mode='val')
    if dataset == 'head':
        train_dataset = VOCDetection(cfg.HEAD.DIR, image_sets=[
                                     ('Part_A', 'trainval'), ('Part_B', 'trainval')],
                                     target_transform=VOCAnnotationTransform(),
                                     mode='train',
                                     dataset_name='VOCPartAB')
        val_dataset = VOCDetection(cfg.HEAD.DIR, image_sets=[('Part_A', 'test'), ('Part_B', 'test')],
                                   target_transform=VOCAnnotationTransform(),
                                   mode='test',
                                   dataset_name='VOCPartAB')
    if dataset == 'crowedhuman':
        train_dataset = ReadDataset(cfg.crowedhuman_train_file,cfg.crowedhuman_dir)
        val_dataset = ReadDataset(cfg.crowedhuman_val_file,cfg.crowedhuman_dir)
    if dataset == 'coco':
        train_dataset = ReadDataset(cfg.coco_train_file,cfg.cocodir)
        val_dataset = ReadDataset(cfg.coco_val_file,cfg.cocodir)
    return train_dataset, val_dataset


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    # impaths = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.from_numpy(sample[1].copy()).float())  #torch.FloatTensor(sample[1]))
        # impaths.append(sample[2])
    return torch.stack(imgs, 0), targets
