# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/11/19 10:09
#project: head detect
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  histogram
####################################################
import sys
import os
import numpy as np 
from matplotlib import pyplot as plt 
import argparse
from collections import defaultdict

def parms():
    parser = argparse.ArgumentParser(description='refinedet traing log')
    parser.add_argument('--file_in', default=None,
                        type=str, help='log file')
    parser.add_argument('--data_name', default='voc', type=str,
                        help='traing data name')
    parser.add_argument('--loss_name',type=str,default='total',help='loss')
    parser.add_argument('--load_num',type=int,default=0,help='load model num')
    parser.add_argument('--file2_in',type=str,default=None,help='load file')
    parser.add_argument('--cmd_type',type=str,default=None,help='')
    return parser.parse_args()

def read_data(file_in):
    '''
    file_in: log file 
        epoch || iter || tloss,confloss,locloss || lr 
    '''
    f_r = open(file_in,'r')
    file_cnts = f_r.readlines()
    epoch_datas = []
    arm_l = []
    arm_c =[]
    total_loss = []
    i = 0
    for tmp in file_cnts:
        i+=1
        tmp_splits = tmp.strip().split('||')
        if len(tmp_splits) <2:
            continue
        splits0 = tmp_splits[1].strip().split(':')
        #print(splits0)
        splits1 = tmp_splits[2].strip().split(',')
        #print(splits1)
        tloss = splits1[0].strip().split(':')[-1]
        if tloss == 'inf':
            continue
        confloss = splits1[1].strip().split(':')[-1]
        locloss = splits1[2].strip().split(':')[-1]
        # epoch_datas.append(int(splits0[1].strip()))
        epoch_datas.append(i)
        total_loss.append(float(tloss))
        arm_l.append(float(locloss))
        arm_c.append(float(confloss))
    loss_datas = [arm_l,arm_c,total_loss]
    return epoch_datas,loss_datas

def plot_lines(txt_path,name):
    ax_data,total_data = read_data(txt_path)
    fig = plt.figure(num=0,figsize=(20,10))
    plt.plot(ax_data,total_data[-1],label='total' )
    plt.plot(ax_data,total_data[0],label='loc')
    plt.plot(ax_data,total_data[1],label='conf')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('%s_training' % name)
    plt.grid(True)
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.2)
    #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    plt.savefig("./logs/%s.png" % name ,format='png')
    plt.show()

def plot_pr(file_in,name):
    '''
    file_in: file record rec and prec 
    '''
    f_r = open(file_in,'r')
    f_cnts = f_r.readlines()
    rec_dict = defaultdict(list)
    total = len(f_cnts)
    for idx,tmp in enumerate(f_cnts):
        if idx == total-1:
            continue
        tmp_splits = tmp.strip().split(',')
        rec = tmp_splits[0].split()[-1]
        prec = float(tmp_splits[1].split()[-1])
        score = float(tmp_splits[2].split()[-1])
        value_ = rec_dict.setdefault(rec,[0,0])
        if value_[0] < prec:
            rec_dict[rec] = [prec,score]
    f_r.close()
    # plot
    rec_keys = rec_dict.keys()
    prec_data = []
    score_data = []
    for tmp_key in rec_keys:
        prec_data.append(rec_dict[tmp_key][0])
        score_data.append(rec_dict[tmp_key][1])
    rec_data = list(map(float,rec_keys))
    ax_data = rec_data
    ay_data = prec_data
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].plot(ax_data,ay_data,label='ROC')
    axs[0].set_ylabel('precision')
    axs[0].set_xlabel('recall')
    axs[0].grid(True)
    axs[0].set_title('ROC')
    axs[1].plot(score_data,ax_data,label='s-r')
    axs[1].set_ylabel('recall')
    axs[1].set_xlabel('score')
    axs[1].grid(True)
    axs[1].set_title('S-R')
    axs[2].plot(score_data,ay_data,label='s-p')
    axs[2].set_ylabel('precision')
    axs[2].set_xlabel('score')
    axs[2].grid(True)
    axs[2].set_title('S-P')
    '''
    plt.plot(ax_data,ay_data,label='ROC')
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title('%s_roc' % name)
    plt.grid(True)
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.2)
    '''
    #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    plt.savefig("../logs/%s.png" % name ,format='png')
    plt.show()

def plot_roces(base_dir):
    '''
    '''
    name_list = ['person']
    for tmp in name_list:
        input_file = os.path.join(base_dir,tmp+'_pr.txt')
        plot_pr(input_file,tmp)

def plt_histgram(file_in,file_out,distance,num_bins=20):
    '''
    file_in: saved train img path  txt file: /img_path/0.jpg  1188
    file_out: output bins and 
    '''
    out_name = file_out
    input_file=open(file_in,'r')
    out_file=open(file_out,'w')
    data_arr=[]
    print(out_name)
    out_list = out_name.strip()
    out_list = out_list.split('/')
    out_name = "./output/"+out_list[-1][:-4]+".png"
    print(out_name)
    id_dict_cnt = dict()
    for line in input_file.readlines():
        line = line.strip()
        line_splits = line.split(' ')
        key_name=int(line_splits[-1])
        cur_cnt = id_dict_cnt.setdefault(key_name,0)
        id_dict_cnt[key_name] = cur_cnt +1
    for key_num in id_dict_cnt.keys():
        data_arr.append(id_dict_cnt[key_num])
    data_in=np.asarray(data_arr)
    if distance is None:
        max_bin = np.max(data_in)
    else:
        max_bin = distance
    datas,bins,c=plt.hist(data_in,num_bins,range=(0.0,max_bin),normed=0,color='blue',cumulative=0)
    #a,b,c=plt.hist(data_in,num_bins,normed=1,color='blue',cumulative=1)
    plt.title('histogram')
    plt.savefig(out_name, format='png')
    plt.show()
    for i in range(num_bins):
        out_file.write(str(datas[i])+'\t'+str(bins[i])+'\n')
    input_file.close()
    out_file.close()

def plot_bar(data_dict,name,data2_dict=None):
    #menMeans = (20, 35, 30, 35, 27)
    #womenMeans = (25, 32, 34, 20, 25)
    #menStd = (2, 3, 4, 1, 2)
    #womenStd = (3, 5, 2, 3, 3)
    keys = data_dict.keys()
    keys = [ '16','32','64', '128', '256','512']
    bar_data = []
    x_labels = keys
    for tmp in keys:
        if data2_dict is None:
            bar_data.append(data_dict[tmp])
        else:
            bar_data.append(data2_dict[tmp]+data_dict[tmp])
    ind = np.arange(len(bar_data))  # the x locations for the groups
    if not isinstance(bar_data,tuple):
        bar_data = tuple(bar_data)    
    if not isinstance(x_labels,tuple):
        x_labels = tuple(x_labels)
    width = 0.35       # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, bar_data, width)
    #p2 = plt.bar(ind, womenMeans, width,
    #            bottom=menMeans, yerr=womenStd)
    #plt.ylabel('bounding-box-num')
    plt.title('%s' % name)
    plt.xticks(ind, x_labels)
    #plt.yticks(np.arange(0, 270000, 10000))
    #plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    outname = '../logs/%s.png' % name
    plt.savefig(outname,format='png')
    plt.show()
    
def get_dataset(f_path):
    f_r = open(f_path,'r')
    f_cnts = f_r.readlines()
    data_dict = dict()
    for tmp in f_cnts:
        tmp_splits = tmp.strip().split(',')
        data_dict[tmp_splits[0]] = int(tmp_splits[1])
    return data_dict

def plt_dataset(f1_path,dataname,f2_path=None):
    data1_dict = get_dataset(f1_path)
    if f2_path is not None:
        data2_dict = get_dataset(f2_path)
    else:
        data2_dict = None
    plot_bar(data1_dict,dataname,data2_dict)

def plot_score_ap(file_in,name):
    '''
    file_in: file record conf_score and ap 
    '''
    f_r = open(file_in,'r')
    f_cnts = f_r.readlines()
    total = len(f_cnts)
    scores = list()
    aps = list()
    for idx,tmp in enumerate(f_cnts):
        if idx == total-1:
            continue
        tmp_splits = tmp.strip().split(',')
        score = float(tmp_splits[0].split()[-1])
        ap = float(tmp_splits[1].split()[-1])
        scores.append(score)
        aps.append(ap)
    f_r.close()
    # plot
    ax_data = scores
    ay_data = aps
    plt.plot(ax_data,ay_data,label='cofidence-ap')
    plt.ylabel('ap')
    plt.xlabel('conf')
    plt.title('%s_conf_ap' % name)
    plt.grid(True)
    #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    #leg.get_frame().set_alpha(0.8)
    #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    plt.savefig("../logs/%s.png" % name ,format='png')
    plt.show()

def get_detect_results(filein,keydata):
    '''
    input: txt file, example: imagepath, bbox_scores
    return: a dict, key in [16, 32, 64, 128, 256, 512] 
    '''
    # read list of images
    keynames = np.array(keydata) #[16, 32, 64, 128, 256, 512] [24,48,96,192,384,640]
    dict_out = defaultdict(list)
    det_read = open(filein, 'r')
    detection_cnts = det_read.readlines()
    det_read.close()
    # image_ids = []
    # boxes = []
    if len(detection_cnts) >=1:
        for tmp_cnt in detection_cnts:
            tmp_splits = tmp_cnt.strip().split(',')
            # image_ids.append(tmp_splits[0])
            confidence = float(tmp_splits[1])
            tmp_box = list(map(float,tmp_splits[2:]))
            tmp_min = min(tmp_box[2]-tmp_box[0],tmp_box[3]-tmp_box[1])
            keep_ids = np.where(keynames >= tmp_min)
            if len(keep_ids[0])>0:
                dict_out[str(keynames[keep_ids[0][0]])].append(confidence)
            else:
                dict_out[str(keynames[-1])].append(confidence)
    return dict_out


def plot_size2score(filein,name):
    '''
    datadict: input data. keys are bounding-box size, values are scores
    '''
    keys = [16,32,64,128,256,512] #[24,48,96,192,384,640]
    datadict = get_detect_results(filein,keys)
    num_bins = 10
    row_num = 2
    col_num = 3
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num)
    for i in range(row_num):
        for j in range(col_num):
            bin_cnt,bin_data,patchs = axes[i,j].hist(datadict[str(keys[i*col_num+j])],num_bins,normed=0,color='blue',cumulative=0) #range=(0.0,max_bin)
            axes[i,j].set_xlabel('score')
            axes[i,j].set_ylabel('num')
            axes[i,j].set_title('size-%s-hist' % str(keys[i*col_num+j]))
            axes[i,j].grid(True)
    plt.savefig('../logs/%s.png' % name,format='png')
    plt.show()

def get_test_gt(filein,keydata):
    '''
    get test or val datas from txt, 
    example: imagename, x1,y1,x2,y2,label, ...
    '''
    # read list of images
    imgset_f = open(filein, 'r')
    lines = imgset_f.readlines()
    imgset_f.close()
    dictdata = defaultdict(lambda:0)
    keynames = np.array(keydata)
    for tmp in lines:
        tmp_splits = tmp.strip().split(',')
        img_name = tmp_splits[0].split('/')[-1][:-4] if len(tmp_splits[0].split('/')) >0 else tmp_splits[0][:-4]
        #img_path = os.path.join(args.img_dir,tmp_splits[0])
        bbox_label = map(float, tmp_splits[1:])
        if not isinstance(bbox_label,list):
            bbox_label = list(bbox_label)
        bbox_label = np.reshape(bbox_label,(-1,5))
        bbox = bbox_label[:,:4]
        for tmp in bbox:
            dmin = min(tmp[2]-tmp[0],tmp[3]-tmp[1])
            keep_idx = np.where(keynames >=dmin)
            if len(keep_idx[0])>0:
                dictdata[str(keynames[keep_idx[0][0]])] +=1
            else:
                dictdata[str(keynames[-1])] +=1
    return dictdata

def plot_gt_size(filein,name):
    '''
    '''
    keys = [16,32,64,128,256,512]
    datadict = get_test_gt(filein,keys)
    plot_bar(datadict,name)

def test_fhog(k):
    w = np.zeros(2*k)
    print(int(k/2))
    for j in range( int(k/2)):
        b_x = k / 2 + j + 0.5
        a_x = k / 2 - j - 0.5
        w[j * 2] = 1.0 / a_x * ((a_x * b_x) / (a_x + b_x))
        w[j * 2 + 1] = 1.0 / b_x * ((a_x * b_x) / (a_x + b_x))
    for j in range(int(k/2),k):
        a_x = j - k / 2 + 0.5
        b_x = -j + k / 2 - 0.5 + k
        w[j * 2] = 1.0 / a_x * ((a_x * b_x) / (a_x + b_x))
        w[j * 2 + 1] = 1.0 / b_x * ((a_x * b_x) / (a_x + b_x))
    ay_data = w
    ax_data = np.arange(2*k)
    plt.plot(ax_data,ay_data,label='test')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.grid(True)
    #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    #leg.get_frame().set_alpha(0.8)
    #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    # plt.savefig("../logs/%s.png" % name ,format='png')
    plt.show()

if __name__=='__main__':
    args = parms()
    file_in = args.file_in
    data_name = args.data_name
    loss_name = args.loss_name
    file2_in = args.file2_in
    if args.cmd_type == 'plot_pr':
        plot_roces(file_in)
    elif args.cmd_type == 'plot_data':
        plt_dataset(file_in,data_name,file2_in)
    elif args.cmd_type == 'plot_trainlog':
        plot_lines(file_in,data_name)
    elif args.cmd_type == 'plot_file':
        plot_pr(file_in,data_name)
    elif args.cmd_type == 'score_ap':
        plot_score_ap(file_in,data_name)
    elif args.cmd_type == 'size_score':
        plot_size2score(file_in,data_name)
    elif args.cmd_type == 'valsize_bar':
        plot_gt_size(file_in,data_name)
    else:
        test_fhog(4)
        print('please input cmd')