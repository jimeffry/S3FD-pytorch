###############################################
#created by :  lxy
#Time:  2020/1/7 10:09
#project: head detect
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  histogram
####################################################
from itertools import product as product
import math
import numpy as np

class PriorBox(object):
    def __init__(self,cfg):
        self.imh = cfg.resize_height
        self.imw = cfg.resize_width
        # number of priors for feature map location (either 4 or 6)
        self.variance = cfg.VARIANCE 
        self.min_sizes = cfg.ANCHOR_SIZES
        self.steps = cfg.STEPS
        self.clip = cfg.CLIP
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

        output = np.array(mean)
        output = np.reshape(output,[-1,4])
        if self.clip:
            output = np.clip(output,0.0,1.0)
        return output

if __name__=='__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
    from config import cfg
    te = PriorBox(cfg)
    s= te()
    print(s.shape)