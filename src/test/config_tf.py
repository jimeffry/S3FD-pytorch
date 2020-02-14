from easydict import EasyDict
_C = EasyDict()
cfg = _C
cfg.vgg_filters = [64,128,256,512]
cfg.extract_filters = [256, 512, 128, 256]
cfg.sfd_fpn_filters_out = [256,512,512,1024,512,256]
#
cfg.NUM_CLASSES=2
_C.resize_width = 640
_C.resize_height = 640
# anchor config
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES = [16, 32, 64, 128, 256, 512]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# detection config
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 1000
_C.TOP_K = 750
_C.CONF_THRESH = 0.1

