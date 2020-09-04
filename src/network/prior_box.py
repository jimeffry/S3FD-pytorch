#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import torch
from itertools import product as product
from math import sqrt 
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self):
        super(PriorBox, self).__init__()
        self.imh = cfg.INPUT_SIZE
        self.imw = cfg.INPUT_SIZE

        # number of priors for feature map location (either 4 or 6)
        self.variance = cfg.VARIANCE or [0.1]
        #self.feature_maps = cfg.FEATURE_MAPS
        self.min_sizes = cfg.ANCHOR_SIZES
        self.steps = cfg.STEPS
        self.clip = cfg.CLIP
        self.aspect_ratios = [0.5,1.0,2.0]
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.feature_maps = cfg.FEATURE_MAPS


    def __call__(self):
        mean = []
        for k in range(len(self.feature_maps)):
            feath = self.feature_maps[k]
            featw = self.feature_maps[k]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh

                mean += [cx, cy, s_kw, s_kh]
                # for ar in self.aspect_ratios:
                    # mean += [cx, cy, s_kw*sqrt(ar), s_kh/sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    # from .configs.config import cfg
    p = PriorBox()
    out = p()
    print(out.size())
