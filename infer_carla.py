"""
Evaluation Scripts
"""
from __future__ import absolute_import, division

import argparse
import logging
import os
import time
from collections import OrderedDict, namedtuple
from glob import glob

import network
import numpy as np
import optimizer
import torch
import torchvision.transforms as standard_transforms
from config import assert_and_infer_cfg, cfg
from datasets.carla_labels import carla_color2oodId
from network import mynn
from ood_metrics import fpr_at_95_tpr
from PIL import Image
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from tqdm import tqdm

dirname = os.path.dirname(__file__)
pretrained_model_path = os.path.join(dirname, 'pretrained/r101_os8_base_cty.pth')

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument(
    '--arch',
    type=str,
    default='network.deepv3.DeepR101V3PlusD_OS8',
    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes', help='possible datasets for statistics; cityscapes')
parser.add_argument('--fp16', action='store_true', default=False, help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int, help='parameter used by apex library')
parser.add_argument('--trunk', type=str, default='resnet101', help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--bs_mult', type=int, default=2, help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1, help='Batch size for Validation per gpu')
parser.add_argument('--class_uniform_pct', type=float, default=0, help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024, help='tile size for class uniform sampling')
parser.add_argument(
    '--batch_weighting',
    action='store_true',
    default=False,
    help='Batch weighting for class (use nll class weighting using batch stats')
parser.add_argument('--jointwtborder', action='store_true', default=False, help='Enable boundary label relaxation')

parser.add_argument('--snapshot', type=str, default=pretrained_model_path)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--date', type=str, default='default', help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default', help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='', help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt', help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb', help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True, help='Use Synchronized BN')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str, help='url used to set up distributed training')
parser.add_argument('--backbone_lr', type=float, default=0.0, help='different learning rate on backbone network')
parser.add_argument('--pooling', type=str, default='mean', help='pooling methods, average is better than max')

parser.add_argument(
    '--ood_dataset_path',
    type=str,
    default='/home/nas1_userB/dataset/ood_segmentation/fishyscapes',
    help='OoD dataset path')

# Anomaly score mode - msp, max_logit, standardized_max_logit
parser.add_argument(
    '--score_mode',
    type=str,
    default='standardized_max_logit',
    help='score mode for anomaly [msp, max_logit, standardized_max_logit]')

# Boundary suppression configs
parser.add_argument('--enable_boundary_suppression', type=bool, default=True, help='enable boundary suppression')
parser.add_argument('--boundary_width', type=int, default=4, help='initial boundary suppression width')
parser.add_argument('--boundary_iteration', type=int, default=4, help='the number of boundary iterations')

# Dilated smoothing configs
parser.add_argument('--enable_dilated_smoothing', type=bool, default=True, help='enable dilated smoothing')
parser.add_argument('--smoothing_kernel_size', type=int, default=7, help='kernel size of dilated smoothing')
parser.add_argument(
    '--smoothing_kernel_dilation', type=int, default=6, help='kernel dilation rate of dilated smoothing')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

args.world_size = 1

print(f'World Size: {args.world_size}')
if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
args.dist_url = args.dist_url + str(8000 + (int(time.time() % 1000)) // 10)

torch.distributed.init_process_group(
    backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank)


def get_net():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)

    net = network.get_net(args, criterion=None, criterion_aux=None)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, None, None, args.snapshot, args.restore_optimizer)
        print(f"Loading completed. Epoch {epoch} and mIoU {mean_iu}")
    else:
        raise ValueError(f"snapshot argument is not set!")

    class_mean = np.load(f'stats/{args.dataset}_mean.npy', allow_pickle=True)
    class_var = np.load(f'stats/{args.dataset}_var.npy', allow_pickle=True)
    net.module.set_statistics(mean=class_mean.item(), var=class_var.item())

    torch.cuda.empty_cache()
    net.eval()

    return net


def preprocess_image(x, mean_std):
    x = Image.fromarray(x)
    x = standard_transforms.ToTensor()(x)
    x = standard_transforms.Normalize(*mean_std)(x)

    x = x.cuda()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return x


if __name__ == '__main__':
    net = get_net()

    mean_std = ([0.4731, 0.4955, 0.5078], [0.2753, 0.2715, 0.2758])
    ood_data_root = args.ood_dataset_path
    glob_path = os.path.join(ood_data_root, '**/rgb_v/*.png')
    image_file_list = glob(glob_path, recursive=True)
    mask_file_list = [i.replace('rgb_v', 'mask_v') for i in image_file_list]

    for image_path, mask_path in zip(tqdm(image_file_list), mask_file_list):

        image = np.array(Image.open(image_path).convert('RGB')).astype('uint8')
        with torch.no_grad():
            image = preprocess_image(image, mean_std)
            main_out, anomaly_score = net(image)
        del main_out

        os.makedirs(os.path.dirname(image_path.replace('rgb_v', 'anomaly_score_v/SML')), exist_ok=True)
        os.makedirs(os.path.dirname(image_path.replace('rgb_v', 'anomaly_vis_v/SML')), exist_ok=True)

        anomaly_score = anomaly_score.cpu().numpy()[0]
        np.save(image_path.replace('rgb_v', 'anomaly_score_v/SML_test').replace('.png', '.npy'), anomaly_score)

        anomaly_score = -(anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min()) * 255
        Image.fromarray(anomaly_score.astype(np.uint8), mode='L').save(image_path.replace('rgb_v', 'anomaly_vis_v/SML'))

