import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import csv


class miniImageNet(Dataset):
    def __init__(self, data_path, split_path, partition='train', transform=None):
        super(Dataset, self).__init__()
        self.data_root = data_path
        self.partition = partition
        self.transform = transform

        file_path = os.path.join(split_path, 'miniImageNet', '{}.csv'.format(self.partition))
        self.imgs, self.labels = self._read_csv(file_path)     

    def _read_csv(self, file_path):
        imgs = []
        labels = []
        labels_name = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i==0:
                    continue
                img, label = row[0], row[1]
                img = os.path.join(self.data_root, 'images/{}'.format(img))
                imgs.append(img)
                if label not in labels_name:
                    labels_name.append(label)
                labels.append(labels_name.index(label))
        return imgs, labels

    def __getitem__(self, item):
        img = self.imgs[item]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[item]
        return img, target

    def __len__(self):
        return len(self.labels)
