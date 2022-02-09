#!/bin/bash
python \
export_onnx.py \
--output-name onnxModel/117_ckpt_dynamic_160_160.onnx \
-f exps/default/yolox_s_l.py \
--image_h  160 \
--image_w  160 \
--batch-size 4 \
--dynamic \
-c YOLOX_outputs/yolox_voc_s_l/late15/117_ckpt.pth