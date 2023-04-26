#!/usr/bin/env bash
# Example on Cityscapes
python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 49494 train.py \
   --dataset carla \
   --val_dataset carla \
   --variant cityscapes/reg-0.3-clip-1000 \
   --arch network.deepv3.DeepR101V3PlusD_OS8 \
   --city_mode 'train' \
   --lr_schedule poly \
   --lr 0.02 \
   --syncbn \
   --poly_exp 0.9 \
   --val_interval 200 \
   --crop_size 768 \
   --scale_min 0.5 \
   --scale_max 2.0 \
   --rrotate 0 \
   --max_iter 60000 \
   --bs_mult 4 \
   --bs_mult_val 1 \
   --gblur \
   --color_aug 0.5 \
   --date 0000 \
   --exp r101_os8_base_60K_cs_enhanced \
   --ckpt ./logs/ \
   --tb_path ./logs/
