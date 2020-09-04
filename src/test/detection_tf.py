###############################################
#created by :  lxy
#Time:  2020/1/7 10:09
#project: head detect
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  histogram
####################################################
import numpy as np

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def nms_py(boxes, scores, threshold=0.7,topk=200,mode='Union'):
    pick = []
    count = 0
    if len(boxes)==0:
        return pick,count
    # print('score',np.shape(scores))
    # boxes = boxes.detach().numpy().copy()
    # scores = scores.detach().numpy().copy()
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # s  = np.array(scores)
    area = np.multiply(x2-x1+1, y2-y1+1)
    ids = np.array(scores.argsort())
    ids = ids[-topk:]
    #I[-1] have hightest prob score, I[0:-1]->others
    while len(ids)>0:
        pick.append(ids[-1])
        xx1 = np.maximum(x1[ids[-1]], x1[ids[0:-1]]) 
        yy1 = np.maximum(y1[ids[-1]], y1[ids[0:-1]])
        xx2 = np.minimum(x2[ids[-1]], x2[ids[0:-1]])
        yy2 = np.minimum(y2[ids[-1]], y2[ids[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == 'Min':
            iou = inter / np.minimum(area[ids[-1]], area[ids[0:-1]])
        else:
            iou = inter / (area[ids[-1]] + area[ids[0:-1]] - inter)
        count +=1
        ids = ids[np.where(iou<threshold)[0]]
        # print(len(ids))
    #result_rectangle = boxes[pick].tolist()
    return pick,count

class Detect_Process(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE
        self.nms_top_k = cfg.NMS_TOP_K

    def __call__(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        num = loc_data.shape[0] 
        num_priors = prior_data.shape[0]
        conf_preds = np.reshape(conf_data,(num, num_priors, self.num_classes))
        conf_preds = np.transpose(conf_preds,(0,2, 1))
        batch_priors = np.reshape(prior_data,(-1, num_priors,4))
        batch_priors = np.tile(batch_priors,reps=(num, 1, 1))
        batch_priors = np.reshape(batch_priors,(-1, 4))
        loc_data = np.reshape(loc_data,(-1,4))
        decoded_boxes = decode(loc_data,batch_priors, self.variance)
        decoded_boxes = np.reshape(decoded_boxes,(num, num_priors, 4))
        output = list()
        for i in range(num):
            boxes = decoded_boxes[i].copy()
            conf_scores = conf_preds[i].copy()
            c_mask = np.where(conf_scores[1] >=self.conf_thresh)
            scores = conf_scores[1][c_mask]
            if len(scores) == 0:
                continue
            boxes_ = boxes[c_mask]
            box_score = [boxes_,scores]
            output.append(box_score)
        return output