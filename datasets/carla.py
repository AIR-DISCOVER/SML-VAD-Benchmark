from torch.utils import data
from PIL import Image
import os
import os.path as osp
import numpy as np
import torch
import torchvision.transforms as transforms
from glob import glob
from datasets.carla_labels import carla_color2trainId, carla_color2oodId, palette

# from config import cfg
# root = cfg.DATASET.CARLA_DIR
root = "/data21/tb5zhh/datasets/new-carla/v3"
num_classes = 19
ignore_label = 255


def file_id(filepath: str):
    # example input: /home/tb5zhh/workspace/2023/SML/SML/data/new-carla/v3/train/seq00-1/rgb_v/1.png
    # example output:seq00-001-001
    return filepath.split("/")[-3].split(
        "-"
    )[0] + "-" + f'{int(filepath.split("/")[-3].split("-")[1]):03d}' + "-" + f'{int(filepath.split("/")[-1][:-4]):03d}'


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new = Image.new("RGB", new_mask.size)
    new = np.array(new)
    for k, v in palette.items():
        new[mask == k] = v
    return Image.fromarray(new)


def make_dataset(mode, variant=None):
    # mode: train/val
    ret_list = []
    mid = 'rgb_v' if mode == 'train' else variant
    for i in sorted(glob(osp.join(root, mode, f"**/{mid}/*.png"), recursive=True), key=lambda i: file_id(i)):
        ret_list.append((i, i.replace(mid, "mask_v")))
    return ret_list


class Carla(data.Dataset):

    def __init__(self,
                 mode,
                 variant,
                 joint_transform=None,
                 transform=None,
                 target_transform=None,
                 target_aux_transform=None,
                 dump_images=False):
        self.mode = mode
        self.variant = variant
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.target_aux_transform = target_aux_transform
        self.dump_images = dump_images

        self.imgs = make_dataset(mode, variant)
        print(f"Found {len(self.imgs)} {mode} images")
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mean_std = ([0.4731, 0.4955, 0.5078], [0.2753, 0.2715, 0.2758])

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]

        input_img = Image.open(img_path)
        img = input_img.copy()
        mask = Image.open(mask_path)

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        trainid_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        oodid_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

        for k, v in carla_color2trainId.items():
            trainid_mask[(mask == k).all(-1)] = v

        for k, v in carla_color2oodId.items():
            oodid_mask[(mask == k).all(-1)] = v

        seg_mask = Image.fromarray(trainid_mask.astype(np.uint8))
        ood_mask = Image.fromarray(oodid_mask.astype(np.uint8))

        # Image Transformations
        if self.joint_transform is not None:
            for idx, xform in enumerate(self.joint_transform):
                img, seg_mask, ood_mask = xform(img, seg_mask, ood_mask)
        if self.transform is not None:
            img = self.transform(img)

        img = transforms.Normalize(*self.mean_std)(img)

        if self.target_aux_transform is not None:
            mask_aux = self.target_aux_transform(seg_mask)
        else:
            mask_aux = torch.tensor([0])
        if self.target_transform is not None:
            seg_mask = self.target_transform(seg_mask)
            ood_mask = self.target_transform(ood_mask)

        # Debug
        if self.dump_images:
            outdir = 'dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            out_mask_fn = os.path.join(outdir, img_name + '_mask.png')
            seg_out_msk_fn = os.path.join(outdir, img_name + '_seg_mask.png')
            ood_out_msk_fn = os.path.join(outdir, img_name + '_ood_mask.png')
            seg_mask_img = colorize_mask(np.array(seg_mask))
            ood_mask_img = colorize_mask(np.array(ood_mask))
            input_img.save(out_img_fn)
            Image.fromarray(mask).save(out_mask_fn)
            seg_mask_img.save(seg_out_msk_fn)
            ood_mask_img.save(ood_out_msk_fn)

        return img, seg_mask, ood_mask, img_name, mask_aux

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataset = Carla("train", dump_images=True)
    print(len(dataset))
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16)
    bincount = np.zeros(256, dtype=np.int64)
    from tqdm import tqdm
    for i in tqdm(dataloader):
        seg = i
        seg = np.bincount(np.array(seg).flatten())
        bincount += seg
        print(bincount[:19])
        print(bincount[-1])
    from IPython import embed; embed()
