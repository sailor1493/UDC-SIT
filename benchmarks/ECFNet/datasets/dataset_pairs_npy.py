import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from functools import partial


def _norm(x, norm=1024):

    return x / norm


def _tonemap(x, k=0.25):
    return x / (x + k)


def set_mapping(mode: str, map_range=1024):
    if mode == "norm":
        print("Registered Normalization")
        return partial(_norm, norm=map_range)
    elif mode == "tonemap":
        print("Registered ToneMap")
        return _tonemap
    else:
        raise NotImplementedError("mode should be norm or tonemap")


def png_loader(path):
    img = Image.open(path)
    return np.array(img, dtype=np.float32)


def setup_load_function(backend: str):
    if backend == "npy":
        return np.load
    elif backend == "png":
        return png_loader


class my_dataset(Dataset):
    def __init__(
        self,
        root_in,
        root_label,
        mapping="norm",
        crop_size=256,
        backend="npy",
        map_range=1024,
    ):
        super(my_dataset, self).__init__()
        self.mapping = set_mapping(mapping, map_range)
        self.load_fn = setup_load_function(backend)

        # in_imgs
        in_files = os.listdir(root_in)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        self.imgs_in.sort()
        # gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]
        self.imgs_gt.sort()

        self.crop_size = crop_size

    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        in_img = self.mapping(self.load_fn(in_img_path))  # Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = self.mapping(self.load_fn(gt_img_path))  # Image.open(gt_img_path)

        data_IN, data_GT = self.train_transform(in_img, gt_img, self.crop_size)
        return data_IN, data_GT

    def train_transform(self, img, label, patch_size=256):
        ih, iw, _ = img.shape
        patch_size = patch_size
        if iw - patch_size > 0:
            ix = random.randrange(0, iw - patch_size)
        else:
            ix = 0
        if ih - patch_size > 0:
            iy = random.randrange(0, ih - patch_size)
        else:
            iy = 0
        # ix = random.randrange(0, max(0, iw - patch_size))
        # iy = random.randrange(0, max(0, ih - patch_size))
        img = img[iy : iy + patch_size, ix : ix + patch_size, :]
        label = label[iy : iy + patch_size, ix : ix + patch_size, :]

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(img)
        label = transform(label)

        return img, label

    def __len__(self):
        return len(self.imgs_in)


class my_dataset_eval(Dataset):
    def __init__(
        self,
        root_in,
        root_label,
        mapping="norm",
        transform=None,
        backend="npy",
        map_range=1024,
    ):
        super(my_dataset_eval, self).__init__()
        self.mapping = self.mapping = set_mapping(mapping, map_range=map_range)
        self.load_fn = setup_load_function(backend)
        # in_imgs
        in_files = os.listdir(root_in)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        self.imgs_in.sort()
        # gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]
        self.imgs_gt.sort()

        self.transform = transform

    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        in_img = self.mapping(self.load_fn(in_img_path))  # Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = self.mapping(self.load_fn(gt_img_path))  # Image.open(gt_img_path)

        img_name = in_img_path.split("/")[-1]

        if self.transform:
            data_IN = self.transform(in_img)
            data_GT = self.transform(gt_img)
        else:
            data_IN = np.asarray(in_img)
            data_IN = torch.from_numpy(data_IN)
            data_IN = data_IN.permute(2, 0, 1)
            data_GT = np.asarray(gt_img)
            data_GT = torch.from_numpy(data_GT)
            data_GT = data_GT.permute(2, 0, 1)
        return data_IN, data_GT, img_name

    def __len__(self):
        return len(self.imgs_in)
