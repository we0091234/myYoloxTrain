#!/bin/bash
python \
export_onnx.py \
--output-name yolox_s_m.onnx \
-f exps/default/yolox_s_m.py \
-c YOLOX_outputs/yolox_voc_s_m/best_ckpt.pth 