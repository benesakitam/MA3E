# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import torch

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import Image, ImageFile
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class MillionAIDDataset(data.Dataset):
    def __init__(self, root, transform=None, rcrop=False):

        print(os.getcwd())

        with open(os.path.join(root, 'random_labels_all.txt'), mode='r') as f:
            train_infos = f.readlines()
        f.close()

        trn_files = []
        trn_targets = []

        for item in train_infos:
            fname, idx = item.strip().split()
            trn_files.append(os.path.join(root + '/test', fname))
            trn_targets.append(int(idx))

        self.files = trn_files
        self.targets = trn_targets

        self.transform = transform
        self.rcrop = rcrop

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]

        img = Image.open(img_path)
        if not self.rcrop:
            img = self.transform(img)
            # todo
            top_start = torch.randperm(14 - 6 + 1)[:1]
            left_start = torch.randperm(14 - 6 + 1)[:1]

            return (img, img, None, top_start, left_start), self.targets[i]
        else:
            img, ori_img, rcrop, top_start, left_start = self.transform(img)
            return (img, ori_img, rcrop, top_start, left_start), self.targets[i]


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
