#!/usr/bin/env bash
# Example on Cityscapes
python -m torch.distributed.launch --nproc_per_node=1 --master_port 44245 calculate_statistics.py \
    --dataset carla \
    --val_dataset carla \
    --arch network.deepv3.DeepR101V3PlusD_OS8 \
    --city_mode 'train' \
    --lr_schedule poly \
    --lr 0.01 \
    --poly_exp 0.9 \
    --snapshot ./pretrained/best_carla_epoch_2_mean-iu_0.68328.pth \
    --crop_size 768 \
    --scale_min 0.5 \
    --scale_max 2.0 \
    --rrotate 0 \
    --max_iter 60000 \
    --bs_mult 4 \
    --gblur \
    --color_aug 0.5 \
    --date 0000 \
    --exp calc_r101_os8_base_60K \
    --ckpt ./logs/ \
    --tb_path ./logs/

