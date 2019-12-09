#!/usr/bin/bash
#CUDA_VISIBLE_DEVICES=0,1  python train.py --dataset head --cuda true --save_folder /wdc/LXY.data/models/head/
#python train.py --lr 0.0004 --dataset crowedhuman --cuda false --save_folder /data/models/head/ --batch_size 2 --multigpu false --resume /data/models/head/sfd_head_60000.pth
#python test/demo.py --modelpath /data/models/head/sfd_crowedhuman_75000.pth --img_dir /data/detect/Scut_Head/SCUT_HEAD_Part_B/JPEGImages --file_in /data/detect/Scut_Head/SCUT_HEAD_Part_B/ImageSets/Main/test.txt
python test/demo.py --modelpath /data/models/head/sfd_crowedhuman_75000.pth  --file_in /home/lxy/Pictures/tests

#*************************test
#CUDA_VISIBLE_DEVICES=0  python tools/eval_voc.py --trained_model /wdc/LXY.data/models/head/sfd_head_60000.pth --save_folder /wdc/LXY.data/head_test --voc_root /wdc/LXY.data/Scut_Head --dataset Part_A
#python tools/plot_pr.py --file-in /data/  --cmd-type plot_pr 
#python tools/plot_pr.py --file_in ./log/2019-11-20-17-01-19.log --data_name head --cmd_type plot_trainlog
