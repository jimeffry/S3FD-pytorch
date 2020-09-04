###############################################
#created by :  lxy
#Time:  2020/1/7 10:09
#project: head detect
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  histogram
####################################################
import os
import cv2
import sys
import numpy as np
import argparse
from tqdm import tqdm
import time
from tensorflow.python.platform import gfile
import tensorflow as tf
from detection_tf import nms_py,Detect_Process
from prior_box_tf import PriorBox
from config_tf import cfg

def parms():
    parser = argparse.ArgumentParser(description='s3df demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.1, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--ctx', default=True, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()

class HeadDetector(object):
    def __init__(self,args):
        self.loadmodel(args.modelpath)
        self.threshold = args.threshold
        self.img_dir = args.img_dir
        
        self.detect = Detect_Process(cfg)
        self.Prior = PriorBox(cfg)
        self.priors =  self.Prior()

    def loadmodel(self,mpath):
        tf_config = tf.ConfigProto()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #tf_config.gpu_options = gpu_options
        tf_config.gpu_options.allow_growth=True  
        tf_config.log_device_placement=False
        self.sess = tf.Session(config=tf_config)
        # self.sess = tf.Session()
        modefile = gfile.FastGFile(mpath, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(modefile.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='headdetect_graph') #return_elements=["images:0","location:0","confidence:0"]) self.input_image,self.loc_out,self.conf_out=
        # self.sess.run(tf.global_variables_initializer())
        # print("************begin to print graph*******************")
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     #if 'images' in m.name or 'location' in m.name or 'confidence' in m.name:
        #     print(m.values()) #m.name,
        # print("********************end***************")
        self.input_image = self.sess.graph.get_tensor_by_name('headdetect_graph/images:0')
        # self.loc = self.sess.graph.get_tensor_by_name('headdetect_graph/location:0')
        # self.conf = self.sess.graph.get_tensor_by_name('headdetect_graph/confidence:0')
        self.loc_out1 = self.sess.graph.get_tensor_by_name('headdetect_graph/location1:0')
        self.loc_out2 = self.sess.graph.get_tensor_by_name('headdetect_graph/location2:0')
        self.loc_out3 = self.sess.graph.get_tensor_by_name('headdetect_graph/location3:0')
        self.loc_out4 = self.sess.graph.get_tensor_by_name('headdetect_graph/location4:0')
        self.loc_out5 = self.sess.graph.get_tensor_by_name('headdetect_graph/location5:0')
        self.loc_out6 = self.sess.graph.get_tensor_by_name('headdetect_graph/location6:0')
        self.conf_out1 = self.sess.graph.get_tensor_by_name('headdetect_graph/confidence1:0')
        self.conf_out2 = self.sess.graph.get_tensor_by_name('headdetect_graph/confidence2:0')
        self.conf_out3 = self.sess.graph.get_tensor_by_name('headdetect_graph/confidence3:0')
        self.conf_out4 = self.sess.graph.get_tensor_by_name('headdetect_graph/confidence4:0')
        self.conf_out5 = self.sess.graph.get_tensor_by_name('headdetect_graph/confidence5:0')
        self.conf_out6 = self.sess.graph.get_tensor_by_name('headdetect_graph/confidence6:0')
        self.net_out = [self.loc_out1,self.loc_out2,self.loc_out3,self.loc_out4,self.loc_out5,self.loc_out6,self.conf_out1,self.conf_out2,self.conf_out3,self.conf_out4,self.conf_out5,self.conf_out6]

    def propress(self,img):
        rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        img = cv2.resize(img,(cfg.resize_width,cfg.resize_height))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img -= rgb_mean
        #img = img[:,:,::-1]
        img = np.transpose(img,(2,0,1))
        return img
        
    def inference_img(self,imgorg):
        t1 = time.time()
        img = self.propress(imgorg.copy())
        bt_img = np.expand_dims(img,0)
        output = self.sess.run([self.loc,self.conf],feed_dict={self.input_image:bt_img})
        t2 = time.time()
        # for tmp in output:
        #     print("debug*********",np.shape(tmp))
        bboxes = self.detect(output[0],output[1],self.priors)
        bboxes = self.nms_filter(bboxes)
        t3 = time.time()
        print('consuming:',t2-t1,t3-t2)
        showimg = self.label_show(bboxes,imgorg)
        return showimg,bboxes

    def propress_imglist(self,imglist):
        rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        outlist = []
        for img in imglist:
            img = cv2.resize(img,(cfg.resize_width,cfg.resize_height))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.astype('float32')
            img -= rgb_mean
            #img = img[:,:,::-1]
            img = np.transpose(img,(2,0,1))
            outlist.append(img)
        return outlist

    def inference_imglist(self,imglist):
        # t1 = time.time()
        batch_num = len(imglist)
        pre_imglist = self.propress_imglist(imglist)
        bt_img = np.array(pre_imglist)
        print("*******input shape:",bt_img.shape)
        output = self.sess.run([self.net_out],feed_dict={self.input_image:bt_img})
        # t2 = time.time()
        # for tmp in output[0]:
            # print("debug*********",np.shape(tmp))
        locations = output[0][:6]
        confidences = output[0][6:]
        max_conf1 = np.max(confidences[0][:,:,:,:3], axis=3,keepdims=True)
        conf1 = np.concatenate((max_conf1, confidences[0][:,:,:,3:]), axis=3)
        locations = np.concatenate([np.reshape(tmp,[batch_num,-1]) for tmp in locations],axis=1)
        confidences = np.concatenate([np.reshape(tmp,[batch_num,-1]) for tmp in confidences[1:]],axis=1)
        confidences = np.concatenate((conf1.reshape(batch_num,-1),confidences),axis=1)
        locations = np.reshape(locations,[batch_num,-1,4])
        confidences = np.reshape(confidences,[batch_num,-1,2])
        confidences = self.softmax(confidences)
        bboxes = self.detect(locations,confidences,self.priors)
        outimgs = []
        for idx,tmp_box in enumerate(bboxes):
            tmp = self.nms_filter([tmp_box])
            # t3 = time.time()
            # print('consuming:',t2-t1,t3-t2)
            showimg = self.label_show(tmp,imglist[idx])
            outimgs.append(showimg)
        return outimgs,bboxes

    def softmax(self,datas):
        datas_u = np.exp(datas)
        datas_d = np.sum(datas_u,axis=2,keepdims=True)
        datas_d = np.concatenate((datas_d,datas_d),axis=2)
        num_a,num_b = datas.shape[:2]
        tmp_z = np.zeros((num_a,num_b,2))
        out = np.where(datas_d< 1e-15,tmp_z,datas_u/datas_d)
        # if datas_d < 1e-15:
        #     num_a,num_b = datas.shape[:2]
        #     out = np.zeros((num_a,num_b,2))
        # else:
        #     out = datas_u/datas_d
        return out

    def nms_filter(self,bboxes):
        scale = np.array([640,640,640,640])[np.newaxis,]
        boxes = bboxes[0][0] * scale
        scores = bboxes[0][1]
        ids, count = nms_py(boxes, scores, 0.2,1000)
        boxes = boxes[ids[:count]]
        # print(np.shape(scores),count)
        scores = scores[ids[:count]]
        boxes_out = []
        for j in range(boxes.shape[0]):
            dets = boxes[j] 
            score = scores[j]
            x1,y1 = dets[:2]
            x2,y2 = dets[2:]
            min_re = min(y2-y1,x2-x1)
            if min_re < 16:
                thresh = 0.6
            else:
                thresh = 0.7
            if score >= thresh:
                boxes_out.append(dets)
        return [[boxes_out,scores]]

    def label_show(self,rectangles,img):
        # rectangles = rectangles.data.cpu().numpy()
        img = cv2.resize(img,(640,640))
        bboxes_score = rectangles[0]
        bboxes = bboxes_score[0]
        scores = bboxes_score[1]
        print('**********head num:---',len(bboxes))
        for j in range(len(bboxes)):
            dets = bboxes[j] 
            score = scores[j]
            x1,y1 = dets[:2]
            x2,y2 = dets[2:]
            # min_re = min(y2-y1,x2-x1)
            # if min_re < 16:
            #     thresh = 0.12
            # else:
            #     thresh = 0.9
            # if score >= thresh:
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            txt = "{:.2f}".format(score)
            point = (int(x1),int(y1-5))
            cv2.putText(img,txt,point,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        return img
    def detectheads(self,imgpath):
        if os.path.isdir(imgpath):
            cnts = os.listdir(imgpath)
            for tmp in cnts:
                tmppath = os.path.join(imgpath,tmp.strip())
                img = cv2.imread(tmppath)
                if img is None:
                    continue
                showimg,_ = self.inference_img(img)
                cv2.imshow('demo',showimg)
                cv2.waitKey(0)
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            # if not os.path.exists(self.save_dir):
            #     os.makedirs(self.save_dir)
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                if len(tmp_file.split(','))>0:
                    tmp_file = tmp_file.split(',')[0]
                if not tmp_file.endswith('jpg'):
                    tmp_file = tmp_file +'.jpeg'
                tmp_path = os.path.join(self.img_dir,tmp_file) 
                if not os.path.exists(tmp_path):
                    print(tmp_path)
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp)
                    continue
                frame,_ = self.inference_img(img)                
                cv2.imshow('result',frame)
                #savepath = os.path.join(self.save_dir,save_name)
                cv2.imwrite('test.jpg',frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            cap = cv2.VideoCapture(imgpath)
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame,_ = self.inference_img(img)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            imglist = [img,img]
            if img is not None:
                # grab next frame
                # update FPS counter
                # frame,bboxes = self.inference_img(img)
                framelist,bboxes = self.inference_imglist(imglist)
                # cv2.imshow('result',frame)
                # key = cv2.waitKey(0) 
                # cv2.imwrite('test30.jpg',frame)
                for tmp in framelist:
                    cv2.imshow('result',tmp)
                    cv2.waitKey(0)
        else:
            print('please input the right img-path')


if __name__ == '__main__':
    args = parms()
    detector = HeadDetector(args)
    imgpath = args.file_in
    detector.detectheads(imgpath)