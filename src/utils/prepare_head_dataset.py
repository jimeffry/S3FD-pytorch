import os
import sys
import cv2
import numpy as np
import tqdm
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



def read_xml_gtbox_and_label(xml_path,keep_difficult=True):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    box_list = []
    cnt_diff = 0
    #print("process image ",img_name)
    for child_of_root in root:
        # print(child_of_root.tag,child_of_root.text)
        if child_of_root.tag == 'object':
            label = None
            head_difficult = 0
            for idx,child_item in enumerate(child_of_root):
                tmp_tag = child_item.tag
                tmp_text = child_item.text 
                #print(idx,tmp_tag)
                
                if tmp_tag == 'name' and tmp_text =='head':
                    label = 1
                if tmp_tag == 'bndbox':
                    for node in child_item:
                        if node.tag == 'xmin':
                            x1 = float(node.text) 
                        if node.tag == 'ymin':
                            y1 = float(node.text) 
                        if node.tag == 'xmax':
                            x2= float(node.text) 
                        if node.tag == 'ymax':
                            y2 = float(node.text) 
                    #assert label is not None, 'label is none, error'
                    if label is not None:
                        box_list.extend([x1,y1,x2,y2,label])
                if tmp_tag == 'difficult':
                    head_difficult = tmp_text
                    cnt_diff+=1
    return  box_list,cnt_diff

def convertVOC2Txt(anodir,txtpath,savetxt):
    train_p = open(txtpath,'r')
    save_p = open(savetxt,'w')
    train_items = train_p.readlines()
    img_cnt = 0
    head_cnt = 0
    difficult_cnt = 0
    total_ = len(train_items)
    for idx,tmp in enumerate(train_items):
        sys.stdout.write("\r> process: %d/%d" %(idx,total_))
        sys.stdout.flush()
        tmp = tmp.strip()
        filename = tmp+'.jpeg'
        anopath = os.path.join(anodir,tmp+'.xml')
        gts,diff_cnt = read_xml_gtbox_and_label(anopath)
        if len(gts)==0:
            print(anopath)
            continue
        img_cnt+=1
        difficult_cnt+=diff_cnt
        cnt_gt = np.reshape(gts,(-1,5))
        head_cnt+=cnt_gt.shape[0]
        gts = map(str,gts)
        gt_str = ','.join(gts)
        save_p.write("{},{}\n".format(filename,gt_str))
    print('total_imgs:',img_cnt)
    print('head_nums:',head_cnt)
    print('difficult:',difficult_cnt)
    train_p.close()
    save_p.close()



if __name__=='__main__':
    # fp = '/data/detect/HollywoodHeads/mov_021_179000.xml'
    # tm = read_xml_gtbox_and_label(fp)
    # print(tm)
    annodir = '/data/detect/HollywoodHeads/Annotations'
    train_txt = '/data/detect/HollywoodHeads/Splits/val.txt'
    save_txt = 'hollywood_val.txt'
    convertVOC2Txt(annodir,train_txt,save_txt)