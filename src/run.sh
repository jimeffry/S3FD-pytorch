#!/usr/bin/bash
# sfd_coco_best.pth trained on coco dataset with orginal sfd net, class is 9
# sfd_coco_best2.pth trained on coco dataset, network is  sfd net and iou branch, class is 9
# sfd_crowedhuman_best.pth trained on crowedhuman dataset, network is  sfd net and iou branch, class is 2
# sfd_crowedhuman_280000.pth trained on crowedhuman dataset, network is  sfd net, class is 2

#CUDA_VISIBLE_DEVICES=0,1  python train.py --dataset head --cuda true --save_folder /wdc/LXY.data/models/head/
# python train/train.py --lr 0.0004 --dataset crowedhuman --cuda false --save_folder /data/models/head/ --batch_size 2 --multigpu false #--resume /data/models/head/sfd_head_60000.pth

#****************************demo
# python test/demo.py --modelpath /data/models/head/sfd_crowedhuman_70000.pth --img_dir /data/detect/Scut_Head/SCUT_HEAD_Part_B/JPEGImages --file_in /data/detect/Scut_Head/SCUT_HEAD_Part_B/ImageSets/Main/test.txt
# python test/demo.py --modelpath /data/models/head/sfd_coco_20000.pth  --file_in /data/detect/al1.png
# python test/demo.py --modelpath /data/models/head/sfd_crowedhuman_280000.pth  --file_in /data/peopleflow.mp4
# python test/demo.py --modelpath /data/models/head/sfd_coco_best.pth  --file_in /data/videos/anshan_crops2/752_5.jpg
# python test/demo.py --modelpath /data/models/head/sfd_coco_best2.pth --file_in /home/lxy/Desktop/imgsegs
# python test/demo.py --modelpath /data/models/head/sfd_coco_best2.pth  --file_in /home/lxy/Develop/git_prj/retinanet-pytorch/data/coco2017val.txt --img_dir  /data/detect/COCO
python test/demo.py --modelpath /data/models/head/sfd_crowedhuman_280000.pth  --file_in ../data/crowedhuman_val.txt --img_dir  /data/detect/head/imgs

##########tensorflow
# python test/demo_tf.py --modelpath ../models/s3fd2.pb  --file_in /data/t2.jpg 
# python test/demo_tf.py --modelpath /data/models/head/sfd_tf.pb  --file_in /data/peopleflow.mp4
# python test/demo_tf.py --modelpath ../models/sfd_tf_b2.pb  --file_in /data/tj1.png 
#*************************test
#CUDA_VISIBLE_DEVICES=0 python test/eval_voc.py --conf_threshold 0.2 --trained_model /mnt/data/LXY.data/models/head/sfd_crowedhuman_110000.pth --save_folder /mnt/data/LXY.data/head_test --dataname crowedhuman --voc_root /mnt/data/LXY.data/crowedhuman/imgs --val_file ../data/crowedhuman_val_256.txt  --use_07 True #1
#*******************************plot
# python test/plot_pr.py --file_in /mnt/data/LXY.data/head_test/person_score.txt --data_name crowedhuman --cmd_type score_ap #2
# python test/plot_pr.py --file_in /mnt/data/LXY.data/head_test/person_pr.txt  --cmd_type plot_file --data_name crowedhuman_score_pr #3
#python test/plot_pr.py --file_in ../data/crowedhuman_val.txt  --cmd_type valsize_bar --data_name valsize2bar
#python test/plot_pr.py --file_in /mnt/data/LXY.data/head_test2/detect_out/det_person.txt  --cmd_type size_score --data_name crowedhuman_size2score
# python utils/plot_size2score.py --conf_thresh 0.001 --file_in /mnt/data/LXY.data/head_test/detect_out/det_person.txt --file2_in ../data/crowedhuman_val_256.txt --data_name crowedhuman_s256 #4
#python utils/plot_size2score.py --conf_thresh 0.001 --file_in /mnt/data/LXY.data/head_test2/detect_out/det_person.txt --file2_in ../data/crowedhuman_val.txt --data_name crowedhuman 
#**************** generate training data
# python utils/load_crowdhuman.py

#**************** convert tr2tf
# python test/tr2tf.py

# **************** pb2pbtxt
# python test/converter.py