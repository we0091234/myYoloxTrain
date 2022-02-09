#!/bin/bash
# python \
# demo_accuracy.py \
# image \
# -f exps/default/yolox_s.py \
# -c M:/Yolox/YOLOX-main/model1/best_ckpt.pth \
# --model_path YOLOX_outputs/yolox_voc_s/85+ \
# --image_path /mnt/EPan/pytorch/yolov3-channel-and-layer-pruning-master/data/dataHz/val/ \
# --excel_path YOLOX_outputs/yolox_voc_s/85+/yoloxOne_384_384_modify.xlsx \
# --conf 0.20 \
# --nms 0.45 \
# --tsize_h  384 \
# --tsize_w 384 \
# --save_result \
# --device gpu


#/mnt/EPan/pytorch/yolov3-channel-and-layer-pruning-master/data/dataHz/val


python \
demo_accuracy.py \
image \
-f exps/default/yolox_s_l.py \
-c M:/Yolox/YOLOX-main/model1/best_ckpt.pth \
--model_path YOLOX_outputs/yolox_voc_s_l/late15 \
--image_path /mnt/Gpan/Mydata/dataTest/headBodyDetect/val/ \
--excel_path YOLOX_outputs/yolox_voc_s_l/late15/yoloxBodyHeadPlate_160_96_modify_yolox_s_l.xlsx \
--conf 0.25 \
--nms 0.45 \
--tsize_h  160 \
--tsize_w 96 \
--save_result \
--device gpu