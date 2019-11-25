import os
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        # print('weight L2:',self.weight.size(),self.weight[0],x.size())
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class VGG(object):
    def __init__(self,batch_norm=False):
        self.layer1 = self.block(2,3,64,maxp=False)
        self.layer2 = self.block(2,64,128)
        self.layer3 = self.block(3,128,256)
        self.layer4 = self.block(3,256,512,mmode=True)
        self.layer5 = self.block(3,512,512)
        self.layer6 = nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6)
        self.layer7 = nn.Conv2d(1024,1024,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)

    def conv2(self,kernel_in,kernel_out,k_size,padd=1,bnorm=False):
        conv = nn.Conv2d(kernel_in,kernel_out,kernel_size=k_size,padding=padd)
        norm = nn.BatchNorm2d(kernel_out)
        relu = nn.ReLU(inplace=True)
        if bnorm :
            layers = [conv,norm,relu]
        else:
            layers = [conv,relu]
        return layers
    def maxpool(self,km_size=2,step=2,mode=False):
        return [nn.MaxPool2d(kernel_size=km_size,stride=step,ceil_mode=mode)]

    def block(self,n,filter_in,filter_out,batch_norm=False,mmode=False,maxp=True):
        layers = []
        if maxp:
            layers.extend(self.maxpool(mode=mmode))
        layers+=(self.conv2(filter_in,filter_out,3,bnorm=batch_norm))
        for i in range(1,n):
            layers.extend(self.conv2(filter_out,filter_out,3,bnorm=batch_norm))
        return layers

    def forward(self):
        layers = []
        layers+=self.layer1
        layers.extend(self.layer2)
        layers.extend(self.layer3)
        layers.extend(self.layer4)
        layers.extend(self.layer5)
        layers.extend([self.maxpool1])
        layers.extend([self.layer6])
        layers.extend([self.relu])
        layers.extend([self.layer7])
        layers.extend([self.relu])
        return layers

class ExtractLayers(object):
    def __init__(self,filter_in):
        self.layer1 = self.conv2(filter_in,256,1,0)
        self.layer2 = self.conv2(256,512,3,1,2)
        self.layer3 = self.conv2(512,128,1,0)
        self.layer4 = self.conv2(128,256,3,1,2)

    def conv2(self,kernel_in,kernel_out,k_size,padd=1,step=1,bnorm=False):
        conv = nn.Conv2d(kernel_in,kernel_out,kernel_size=k_size,padding=padd,stride=step)
        norm = nn.BatchNorm2d(kernel_out)
        relu = nn.ReLU(inplace=True)
        if bnorm :
            layers = [conv,norm,relu]
        else:
            layers = [conv,relu]
        return layers
    def forward(self):
        layers = []
        layers += self.layer1
        layers.extend(self.layer2)
        layers.extend(self.layer3)
        layers.extend(self.layer4)
        return layers

class S3FD(nn.Module):
    def __init__(self,cfg):
        super(S3FD,self).__init__()
        #self.lo = nn.Sequential(*vgg(cfg,3))
        net = VGG()
        add_layers = ExtractLayers(1024) 
        self.vgg = nn.ModuleList(net.forward())
        self.extracts = nn.ModuleList(add_layers.forward())
        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)
        #self.extracts = nn.ModuleList(add_extras(extras_cfg,1024))
    def forward(self,x):
        fpn = list()
        loc = list()
        conf = list()
        for idx in range(16):
            x = self.vgg[idx](x)
        s = self.L2Norm3_3(x)
        fpn.append(s)
        for idx in range(16,23):
            x = self.vgg[idx](x)
        s = self.L2Norm4_3(x)
        fpn.append(s)
        for idx in range(23,30):
            x = self.vgg[idx](x)
        s = self.L2Norm5_3(x)
        fpn.append(s)
        for idx in range(30,len(self.vgg)):
            x = self.vgg[idx](x)
        fpn.append(x)
        for idx,tmp in enumerate(self.extracts):
            x = tmp(x)
            if idx == 2 or idx==6:
                fpn.append(x)
        return x

if __name__=="__main__":
    a = torch.ones([1,3,640,640])
    net = S3FD(vgg_cfg)
    print(net)
    c= net(a)