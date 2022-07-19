import random
from PIL import ImageFilter

import torch
from torchvision import transforms

from .rand_augmentation import rand_augment_transform

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def build_transform(args):

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(
        translate_const=int(args.size * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        rand_augment_transform('rand-n2-m10-mstd0.5', ra_params, use_cmc=False),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    print('train transform: ', train_transform)

    return TwoCropTransform(train_transform)