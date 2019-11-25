from easydict import EasyDict
cfg = EasyDict()
cfg.vgg_filters = [64,128,256,512]
cfg.extract_filters = [256, 512, 128, 256]
cfg.sfd_fpn_filters_out = [256,512,512,1024,512,256]
