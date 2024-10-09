# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch
import math
import copy

from torchvision import datasets, transforms
import torchvision.transforms.functional as F


class RandomCropWithPosition(transforms.RandomCrop):
    def __init__(self, coord, size):
        self.coord = coord
        self.size = size

    @staticmethod
    def get_params(img, output_size, *args, **kwargs):  # img: 224 output_size: 96
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[1], img.shape[2]
        th, tw = output_size
        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")
        if w == tw and h == th:
            return 0, 0, h, w

        coord = args[0]
        idx_i = torch.randint(0, coord.shape[0], size=(1,))
        idx_j = torch.randint(0, coord.shape[0], size=(1,))
        i = coord[idx_i].item()
        j = coord[idx_j].item()

        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.get_params(img, (self.size, self.size), self.coord)

        return F.crop(img, i, j, h, w), i, j, h, w


class ScalingCenterCrop(object):
    def __init__(self, input_size, crop_size, nums_crop, r_range):

        self.nums_crop = nums_crop
        self.inp = input_size
        self.crop = crop_size
        self.angles = r_range
        self.bounding = self.crop * (2 ** 0.5)

        self.patch_size = 16
        coord_init = torch.arange(0, self.inp - self.crop, self.patch_size) - ((self.bounding - self.crop) // 2)
        # self.coord = coord_init[torch.gt(coord_init, 0)].int()
        self.coord = coord_init.int()

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.inp, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.crop_bounding = RandomCropWithPosition(self.coord, size=math.ceil(self.bounding))
        self.rotated_crop_t = transforms.Compose([
            transforms.RandomRotation(self.angles),
            transforms.CenterCrop(size=self.crop)
        ])

    def generate_coords(self):
        top_init = torch.arange(0, self.inp, self.patch_size) - ((self.bounding - self.crop) // 2)
        top_init = top_init[torch.gt(top_init, 0)].int()
        left_init = torch.arange(0, self.inp, self.patch_size) - ((self.bounding - self.crop) // 2)
        left_init = left_init[torch.gt(left_init, 0)].int()
        grid_top, grid_left = torch.meshgrid(top_init, left_init)
        coords = torch.stack((grid_top, grid_left), dim=-1).view(-1, 2)

        return coords

    def __call__(self, image):
        img = self.transform(image)
        img_ori = copy.deepcopy(img)

        top_start = []
        left_start = []
        for _ in range(self.nums_crop):
            crop_big, top, left, _, _ = self.crop_bounding(img)
            crop = self.rotated_crop_t(crop_big)
            correct_top = int(top + (self.bounding - self.crop) // 2)
            correct_left = int(left + (self.bounding - self.crop) // 2)
            img[:, correct_top:(correct_top + self.crop), correct_left:(correct_left + self.crop)] = crop

            top_start.append(correct_top // self.patch_size)
            left_start.append(correct_left // self.patch_size)

        return img, img_ori, crop, torch.tensor(top_start), torch.tensor(left_start)
