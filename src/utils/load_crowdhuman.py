import numpy as np 
import os
import shutil
import json
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from collections import defaultdict
import tqdm

def load_file(fpath):#fpath是具体的文件 ，作用：#str to list
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records

def crowdhuman2coco(odgt_path,json_path):#一个输入文件路径，一个输出文件路径
    records = load_file(odgt_path)   #提取odgt文件数据
    #预处理
    json_dict = {"images":[], "annotations": [], "categories": []}#定义一个字典，coco数据集标注格式
    START_B_BOX_ID = 1 #设定框的起始ID
    image_id = 1  #每个image的ID唯一，自己设定start，每次++
    bbox_id = START_B_BOX_ID
    image = {} #定义一个字典，记录image
    annotation = {} #记录annotation
    categories = {}  #进行类别记录
    record_list = len(records)  #获得record的长度，循环遍历所有数据。
    print(record_list)
    #一行一行的处理。
    for i in range(record_list):
        file_name = records[i]['ID']+'.jpg'  #这里是字符串格式  eg.273278,600e5000db6370fb
        #image_id = int(records[i]['ID'].split(",")[0]) 这样会导致id唯一，要自己设定
        im = Image.open("../mmdetection/data/crowdhuman/Images/"+file_name)
        #根据文件名，获取图片，这样可以获取到图片的宽高等信息。因为再odgt数据集里，没有宽高的字段信息。
        image = {'file_name': file_name, 'height': im.size[1], 'width': im.size[0],'id':image_id} #im.size[0]，im.size[1]分别是宽高
        json_dict['images'].append(image) #这一步完成一行数据到字典images的转换。

        gt_box = records[i]['gtboxes']  
        gt_box_len = len(gt_box) #每一个字典gtboxes里，也有好几个记录，分别提取记录。
        for j in range(gt_box_len):
            category = gt_box[j]['tag']
            if category not in categories:  #该类型不在categories，就添加上去
                new_id = len(categories) + 1 #ID递增
                categories[category] = new_id
            category_id = categories[category]  #重新获取它的类别ID
            fbox = gt_box[j]['fbox']  #获得全身框
            #对ignore进行处理，ignore有时在key：extra里，有时在key：head_attr里。属于互斥的。
            ignore = 0 #下面key中都没有ignore时，就设为0，据观察，都存在，只是存在哪个字典里，需要判断一下
            if "ignore" in gt_box[j]['head_attr']:
                ignore = gt_box[j]['head_attr']['ignore']
            if "ignore" in gt_box[j]['extra']:
                ignore = gt_box[j]['extra']['ignore']
            #对字典annotation进行设值。
            annotation = {'area': fbox[2]*fbox[3], 'iscrowd': ignore, 'image_id':  #添加hbox、vbox字段。
                        image_id, 'bbox':fbox, 'hbox':gt_box[j]['hbox'],'vbox':gt_box[j]['vbox'],
                        'category_id': category_id,'id': bbox_id,'ignore': ignore,'segmentation': []}  
            #area的值，暂且就是fbox的宽高相乘了，观察里面的数据，发现fbox[2]小、fbox[3]很大，刚好一个全身框的宽很小，高就很大。（猜测），不是的话，再自行修改
            #segmentation怎么处理？博主自己也不知道，找不到对应的数据，这里就暂且不处理。
            #hbox、vbox、ignore是添加上去的，以防有需要。
            json_dict['annotations'].append(annotation)
            
            bbox_id += 1 #框ID ++
        image_id += 1 #这个image_id的递增操作，注意位置，博主一开始，放上面执行了，后面出了bug，自己可以理一下。
        #annotations的转化结束。
    #下面这一步，对所有数据，只需执行一次，也就是对categories里的类别进行统计。
    for cate, cid in categories.items():
            #dict.items()返回列表list的所有列表项，形如这样的二元组list：［(key,value),(key,value),...］
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)
    #到此，json_dict的转化全部完成，对于其他的key，
    #因为没有用到（不访问），就不需要给他们空间，也不需要去处理，字典是按key访问的，如果自己需要就自己添加上去就行
    json_fp = open(json_path, 'w')
    json_str = json.dumps(json_dict) #写json文件。
    json_fp.write(json_str)
    json_fp.close()

def savetxt(fpath,outpath,basedir):
    records = load_file(fpath)
    total_ims = len(records)
    fw = open(outpath,'w')
    val_cnt = 0
    per_cnt = 0
    img_size_dict = defaultdict(lambda:0)
    img_size_list = []
    for idx in tqdm.tqdm(range(total_ims)):
        tmp_dict = records[idx]
        filename = tmp_dict['ID']+'.jpg'
        filename_s = filename.split(',')
        filename = filename_s[0] +'/'+filename_s[1]
        #print(filename)
        gtboxes = tmp_dict['gtboxes']
        tmp_cnt = len(gtboxes)
        bbox_list = []
        imgpath = os.path.join(basedir,filename)
        im = Image.open(imgpath)
        keyname = min(im.size[0],im.size[1])
        img_size_dict[str(keyname)] +=1
        if keyname > 4000:
            keyname = 4000
        # img_size_list.append(keyname)
        # img = cv2.imread(imgpath)
        for at_id in range(tmp_cnt):
            cnt_dict = gtboxes[at_id]
            if "ignore" in cnt_dict['head_attr']:
                ignore = cnt_dict['head_attr']['ignore']
            if "ignore" in cnt_dict['extra']:
                ignore = cnt_dict['extra']['ignore']
            if int(ignore) == 1:
                # show_box = cnt_dict['hbox']
                # #print(show_box)
                # x1,y1 = int(show_box[0]),int(show_box[1])
                # x2,y2 = int(show_box[0]+show_box[2]),int(show_box[1]+show_box[3])
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0))
                # cv2.imshow('sc',img)
                # cv2.waitKey(100)
                continue
            else:
                tmp_box = cnt_dict['hbox']
                if tmp_box[0] <0:
                    tmp_box[0] = 0
                if tmp_box[1]<0:
                    tmp_box[1] = 0
                if tmp_box[2]<0 or tmp_box[3]<0:
                    continue
                if tmp_box[3]*tmp_box[2] < 256:
                    continue
                if float(tmp_box[3])/float(tmp_box[2]) >2 or float(tmp_box[2])/float(tmp_box[3])>2:
                    pass
                else:
                    tmp_box[2] = tmp_box[2]+tmp_box[0]
                    tmp_box[3] = tmp_box[3]+tmp_box[1]
                    tmp_box.extend([1])
                    bbox_list.extend(tmp_box)
                    per_cnt +=1
        if len(bbox_list)>=100:
            img_size_list.append(keyname)
        if len(bbox_list)>0:
            val_cnt +=1
            bbox_list = list(map(str,bbox_list))
            tmp_str = ','.join(bbox_list)
            fw.write("{},{}\n".format(filename,tmp_str))
    print('dataset:',total_ims)
    print('valid img:',val_cnt)
    print('person identity:',per_cnt)
    # plothist(img_size_list)
    
def plothist(datadict):
    # xdata = datadict.keys()
    # ydata = []
    # for tmp in xdata:
    #     ydata.append(datadict[tmp])
    print('total plt:',len(datadict))
    fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
    # ax.bar(xdata,ydata)
    xn,bd,paths = ax.hist(datadict,bins=50)
    fw = open('../data/crowdhead_imgsize_100val.txt','w')
    for idx,tmp in enumerate(xn):
        fw.write("{}:{}\n".format(tmp,bd[idx]))
    fw.close()
    plt.savefig('../data/crowdhead_100val.png',format='png')
    plt.show()
def copyimg(orgdir,disdir):
    cnts = os.listdir(orgdir)
    print('total',len(cnts))
    for tmp in cnts:
        tmp_s = tmp.strip().split(',')
        dispath = os.path.join(disdir,tmp_s[0])
        if not os.path.exists(dispath):
            os.makedirs(dispath)
        savename = os.path.join(dispath,tmp_s[-1])
        orgname = os.path.join(orgdir,tmp)
        shutil.copyfile(orgname,savename)

def check_widerface(imgdir):
    img_size_list = []
    for dir1 in os.listdir(imgdir):
        tmp_dir = os.path.join(imgdir,dir1.strip())
        for dir2 in os.listdir(tmp_dir):
            imgpath = os.path.join(tmp_dir,dir2.strip())
            img = Image.open(imgpath)
            keyname = min(img.size[0],img.size[1])
            img_size_list.append(keyname)
    plothist(img_size_list)


if __name__=='__main__':
    savetxt('/data/detect/head/annotation_val.odgt','../data/crowedhuman_val_256.txt','/data/detect/head/imgs')
    #copyimg('/wdc/LXY.data/heads/Images','/wdc/LXY.data/heads/imgs')
    #check_widerface('/mnt/data/LXY.data/WIDER_train')