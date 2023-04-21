#!/usr/bin/env bash
# Example on Cityscapes
python train.py \
   --dataset carla \
   --val_dataset carla \
   --arch network.deepv3.DeepR101V3PlusD_OS8 \
   --city_mode 'train' \
   --lr_schedule poly \
   --lr 0.01 \
   --poly_exp 0.9 \
   --val_interval 200 \
   --crop_size 768 \
   --scale_min 0.5 \
   --scale_max 2.0 \
   --rrotate 0 \
   --max_iter 60000 \
   --bs_mult 2 \
   --gblur \
   --color_aug 0.5 \
   --date 0000 \
   --exp r101_os8_base_60K \
   --ckpt ./logs/ \
   --tb_path ./logs/
