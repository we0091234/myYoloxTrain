#!/bin/bash
python \
demo_accuracy.py \
image \
-f exps/default/yolox_s_m.py \
-c M:/Yolox/YOLOX-main/model1/best_ckpt.pth \
--model_path /mnt/Gpan/Mydata/pytorchPorject/yoloxNew/myYoloxTrain/YOLOX_outputs/yolox_voc_s_m \
--image_path /mnt/Gpan/Mydata/dataTest/headBodyDetect/modify0/ \
--excel_path /mnt/Gpan/Mydata/pytorchPorject/yoloxNew/myYoloxTrain/YOLOX_outputs/yolox_voc_s_m/yoloxBodyHeadPlate_modify0.xlsx \
--conf 0.25 \
--nms 0.45 \
--tsize_h  160 \
--tsize_w 160 \
--save_result \
--device gpu
