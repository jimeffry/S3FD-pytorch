import os
import sys
import cv2
import torch
import onnx
from onnx_tf.backend import prepare
# import tensorflow as tf
import numpy as np
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
from vgg16 import S3FD

def rename_dict(state_dict):
    state_dict_new = dict()
    for key,value in list(state_dict.items()):
        state_dict_new[key[7:]] = value
    return state_dict_new

def tr2onnx(modelpath):
    # Load the trained model from file
    device = 'cpu'
    # net = shufflenet_v2_x1_0(pretrained=False,num_classes=6)
    net = S3FD(2)
    state_dict = torch.load(modelpath,map_location=device)
    # state_dict = rename_dict(state_dict)
    net.load_state_dict(state_dict)
    net.eval()
    # Export the trained model to ONNX
    dummy_input = Variable(torch.randn(1, 640,640,3)) # picture will be the input to the model
    export_onnx_file = '../models/s3fd2.onnx'
    torch.onnx.export(net,
                    dummy_input,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True, # 是否执行常量折叠优化
                    input_names=["images"], # 输入名
                    output_names=["location1","location2","location3","location4","location5","location6","confidence1","confidence2","confidence3","confidence4","confidence5","confidence6",], # 输出名
                    # dynamic_axes={"images":{0:"batch_size"}, # 批处理变量
                    #                 "location":{0:"batch_size"},
                    #                 "confidence":{0:"batch_size"}}
    )

def onnx2tf(modelpath):
    # Load the ONNX file
    model = onnx.load(modelpath)
    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model)
    # Input nodes to the model
    print('inputs:', tf_rep.inputs)
    # Output nodes from the model
    print('outputs:', tf_rep.outputs)
    # All nodes in the model
    print('tensor_dict:')
    # print(tf_rep.tensor_dict)
    # 运行tensorflow模型
    print('Image 1:')
    # img = cv2.imread('/data/detect/shang_crowed/part_B_final/test_data/images/IMG_2.jpg')
    # img = cv2.resize(img,(1920,1080))
    # img = np.transpose(img,(2,0,1))
    # output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis,:,:, :])
    # print('The digit is classified as ', np.sum(output))
    tf_rep.export_graph('../models/s3fd2.pb')


if __name__=='__main__':
    modelpath = '/data/models/head/sfd_crowedhuman_280000.pth'
    # modelpath = './srd_tr.pth'
    tr2onnx(modelpath)
    modelpath = '../models/s3fd2.onnx'
    onnx2tf(modelpath)