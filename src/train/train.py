#-*- coding:utf-8 -*-
'''
* version_1: follow the paper setting, positive anchors  Iou > 0.35 && 0.1< Iou <0.35,lambda=1
* version_2: positive anchors  Iou > 0.35 && 0.1< Iou <0.35, lambda=4
* version_3: positive anchors Iou > 0.5 && 0.35 < Iou < 0.5, lambda=4
* version_4: positive anchors Iou > 0.35, lambda=4
* version 5: positive anchors Iou > 0.35, lambda=4, && 0.1<Iou<0.35,take the ave_pnum for every gt face,
'''

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import cv2
import time
import torch
import argparse
import collections
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
# from s3fd import S3FD
from vgg16 import S3FD
from prior_box import PriorBox
sys.path.append(os.path.join(os.path.dirname(__file__),'../losses'))
from multibox_loss import MultiBoxLoss
sys.path.append(os.path.join(os.path.dirname(__file__),'../preparedata'))
from factory import dataset_factory, detection_collate
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from bbox_utils import match,match_ssd

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def params():
    parser = argparse.ArgumentParser(
        description='S3FD face Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset',default='face',choices=['hand', 'face', 'coco','head','crowedhuman'],help='Train target')
    parser.add_argument('--basenet',default='vgg16_reducedfc.pth',help='Pretrained base model')
    parser.add_argument('--batch_size',default=2, type=int,help='Batch size for training')
    parser.add_argument('--resume',default=None, type=str,help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers',default=4, type=int,help='Number of workers used in dataloading')
    parser.add_argument('--cuda',default=True, type=str2bool,help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate',default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum',default=0.9, type=float,help='Momentum value for optim')
    parser.add_argument('--weight_decay',default=5e-4, type=float,help='Weight decay for SGD')
    parser.add_argument('--gamma',default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--multigpu',default=False, type=str2bool,help='Use mutil Gpu training')
    parser.add_argument('--save_folder',default='weights/',help='Directory for saving checkpoint models')
    parser.add_argument('--log_dir',default='../logs',help='Directory for saving logs')
    parser.add_argument('--start_epoch',type=int,default=1,help='')
    return parser.parse_args()

def train_net(args):
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    #*******load data
    train_dataset, val_dataset = dataset_factory(args.dataset)
    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True,
                                collate_fn=detection_collate,
                                pin_memory=True)
    val_batchsize = args.batch_size // 2
    val_loader = data.DataLoader(val_dataset, val_batchsize,
                                num_workers=args.num_workers,
                                shuffle=False,
                                collate_fn=detection_collate,
                                pin_memory=True)
    
    # s3fd_net = build_s3fd('train', cfg.NUM_CLASSES)
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    s3fd_net = S3FD(cfg.NUM_CLASSES,cfg.NumAnchor).to(device)
    #print(">>",net)
    if True: #not args.resume:
        print('Initializing weights...')
        s3fd_net.extras.apply(s3fd_net.weights_init)
        s3fd_net.loc.apply(s3fd_net.weights_init)
        s3fd_net.conf.apply(s3fd_net.weights_init)
        s3fd_net.iou.apply(s3fd_net.weights_init)
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        # start_epoch = s3fd_net.load_weights(args.resume)
        s3fd_net.load_state_dict(torch.load(args.resume,map_location=device),strict=False)
    else:
        vgg_weights = torch.load(os.path.join(args.save_folder,args.basenet),map_location=device)
        print('Load base network....')
        s3fd_net.vgg.load_state_dict(vgg_weights)
    if args.multigpu:
        s3fd_net = torch.nn.DataParallel(s3fd_net)
        cudnn.benckmark = True

    optimizer = optim.SGD(s3fd_net.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)
    print('Using the specified args:')
    print(args)
    return s3fd_net,optimizer,criterion,train_loader,val_loader

def createlogger(lpath):
    if not os.path.exists(lpath):
        os.makedirs(lpath)
    logger = logging.getLogger()
    logname= time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'
    logpath = os.path.join(lpath,logname)
    hdlr = logging.FileHandler(logpath)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    return logger

def main():
    args = params()
    logger = createlogger(args.log_dir)
    net,optimizer,criterion,train_loader,val_loader = train_net(args)
    start_epoch = args.start_epoch
    iteration = 0
    net.train()
    rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
    loss_hist = collections.deque(maxlen=200)
    loss_class_min = 10.0
    loss_reg_min = 10.0
    prior_box = PriorBox()
    with torch.no_grad():
        priors =  prior_box()
    for epoch in range(start_epoch, cfg.EPOCHES):
        #losses = 0
        lr = poly_lr_scheduler(optimizer,args.lr,epoch,max_iter=cfg.EPOCHES)
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.cuda:
                images = images.cuda() #Variable(images.cuda())
                targets = [ann.cuda() for ann in targets]
            '''
            conf_t = test_anchor(targets,priors,cfg)
            images = images.cpu().numpy()
            for i in range(args.batch_size):
                tmp_img = np.transpose(images[i],(1,2,0))
                tmp_img = tmp_img + rgb_mean
                #tmp_img = tmp_img * 255
                tmp_img = np.array(tmp_img,dtype=np.uint8)
                tmp_img = cv2.cvtColor(tmp_img,cv2.COLOR_RGB2BGR)
                h,w = tmp_img.shape[:2]
                if len(targets[i])>0:
                    gt = targets[i].cpu().numpy()
                    for j in range(gt.shape[0]):
                        x1,y1 = int(gt[j,0]*w),int(gt[j,1]*h)
                        x2,y2 = int(gt[j,2]*w),int(gt[j,3]*h)
                        # print('pred',x1,y1,x2,y2,gt[j,4],w,h)
                        if x2 >x1 and y2 >y1:
                            cv2.rectangle(tmp_img,(x1,y1),(x2,y2),(0,0,255))
                for j in range(priors.size(0)):
                    if conf_t[i,j] >0:
                        box = priors[j].cpu().numpy()
                        # print(box)
                        x1,y1 = box[:2] - box[2:] / 2
                        x2,y2 = box[:2] + box[2:] / 2
                        x1,y1 = int(x1*w),int(y1*h)
                        x2,y2 = int(x2*w),int(y2*h)
                        cv2.rectangle(tmp_img,(x1,y1),(x2,y2),(255,0,0))
                cv2.imshow('src',tmp_img)
                cv2.waitKey(0)
            '''
            # if iteration in cfg.LR_STEPS:
            #     step_index += 1
            #     adjust_learning_rate(args.lr,optimizer, args.gamma, step_index)
            # t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            # loss_l, loss_c,loss_iou = criterion(out,priors, targets)
            loss_l, loss_c = criterion(out,priors, targets)
            loss = loss_l +  loss_c 
            loss.backward()
            optimizer.step()
            # t1 = time.time()
            loss_hist.append(float(loss.item()))
            if iteration % 100 == 0:
                #tloss = losses / 100.0
                #print('tl',loss.data,tloss)
                logger.info('epoch:{} || iter:{} || tloss:{:.4f}, confloss:{:.4f}, locloss:{:.4f} || lr:{:.6f}'.format(epoch,iteration,np.mean(loss_hist),loss_c.item(),loss_l.item(),lr))
                #losses = 0
            if iteration != 0 and iteration % 100 == 0:
                tmpl,tmpc = val(args,net,val_loader,criterion,priors,logger)
                if tmpl < loss_reg_min or tmpc<loss_class_min:
                    loss_reg_min = tmpl
                    loss_class_min = tmpc
                    logger.info('Saving state, iter: %d' % iteration)
                    sfile = 'sfd_' + args.dataset + '_best.pth'
                    spath = os.path.join(args.save_folder, sfile)
                    if args.multigpu:
                        torch.save(net.module.state_dict(),spath)
                    else:
                        torch.save(net.state_dict(),spath)
            iteration += 1
        #val(args,net,val_loader,criterion)
        if iteration == cfg.MAX_STEPS:
            break
    # torch.save(net.state_dict(),os.path.join(args.save_folder,'sfd_'+args.dataset+'_final.pth'))
    
def val(args,net,val_loader,criterion,priors,logger):
    net.eval()
    lossc_hist = list()
    lossl_hist = list()
    for batch_idx, (images, targets) in enumerate(val_loader):
        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        out = net(images)
        loss_l, loss_c = criterion(out,priors, targets)
        # loss = loss_l + loss_c
        # if loss.item()<100:
        lossc_hist.append(loss_c.item())
        lossl_hist.append(loss_l.item())
    logger.info('val Loss_cls: {:.6f} and Loss_reg: {:.6f}'.format(np.mean(lossc_hist),np.mean(lossl_hist)))
    return np.mean(lossl_hist),np.mean(lossc_hist)

def test_anchor(targets,priors,cfg):
    num_priors = priors.size(0)
    num = len(targets)
    loc_t = torch.Tensor(num, num_priors, 4)
    conf_t = torch.LongTensor(num, num_priors)
    soft_t = torch.Tensor(num,num_priors)
    defaults = priors.data
    for idx in range(num):
        truths = targets[idx][:, :-1].data
        labels = targets[idx][:, -1].data
        match(cfg.HEAD.OVERLAP_THRESH, truths, defaults,cfg.VARIANCE, labels,loc_t, conf_t,soft_t, idx)
        # match_ssd([0.3,0.5], truths, defaults, cfg.VARIANCE, labels, loc_t, conf_t, idx)
    return conf_t


def adjust_learning_rate(init_lr,optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = init_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    #     return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
