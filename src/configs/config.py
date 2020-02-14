from easydict import EasyDict
_C = EasyDict()
cfg = _C
cfg.vgg_filters = [64,128,256,512]
cfg.extract_filters = [256, 512, 128, 256]
cfg.sfd_fpn_filters_out = [256,512,512,1024,512,256]
#
cfg.NUM_CLASSES=2
cfg.INPUT_SIZE = 640
#

_C.resize_width = 640
_C.resize_height = 640
_C.scale = 1 / 127.0
_C.anchor_sampling = True
_C.filter_min_face = True

# train config
#_C.LR_STEPS = (120, 198, 250)
_C.MAX_STEPS = 200000
_C.LR_STEPS = (800000,1000000,1200000)
_C.EPOCHES = 600

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
_C.CONF_THRESH = 0.005

# loss config
_C.NEG_POS_RATIOS = 3
_C.NUM_CLASSES = 2
_C.USE_NMS = True

# hand config
_C.HAND = EasyDict()
_C.HAND.TRAIN_FILE = './data/hand_train.txt'
_C.HAND.VAL_FILE = './data/hand_val.txt'
_C.HAND.DIR = '/home/data/lj/egohands/'
_C.HAND.OVERLAP_THRESH = 0.35

# face config
_C.FACE = EasyDict()
_C.FACE.TRAIN_FILE = './data/face_train.txt'
_C.FACE.VAL_FILE = './data/face_val.txt'
_C.FACE.FDDB_DIR = '/home/data/lj/FDDB'
_C.FACE.WIDER_DIR = '/home/data/lj/WIDER'
_C.FACE.AFW_DIR = '/home/data/lj/AFW'
_C.FACE.PASCAL_DIR = '/home/data/lj/PASCAL_FACE'
_C.FACE.OVERLAP_THRESH = [0.1, 0.35, 0.5]

# head config
_C.HEAD = EasyDict()
_C.HEAD.DIR = '/data/detect/Scut_Head/'
_C.HEAD.OVERLAP_THRESH = [0.1, 0.35, 0.5]
# crowedhuman
_C.crowedhuman_train_file = '../data/crowedhuman_train.txt'
_C.crowedhuman_val_file = '../data/crowedhuman_val.txt'
_C.crowedhuman_dir = '/data/detect/head/imgs'
