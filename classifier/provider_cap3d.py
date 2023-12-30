import os
import cv2
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import sqlitedict

import kiui
from kiui.op import safe_normalize

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Cap3DDataset(Dataset):
    def __init__(self, path='data_cap3d', training=True, split_ratio=0.8):

        self.path = path
        self.training = training

        self.items = []
        self.labels = []

        db = sqlitedict.SqliteDict('objaverse_kiui.sqlite')
        for uid, score in db.items():
            self.items.append(uid)
            self.labels.append(score) # 0/1

        num_train = int(len(self.items) * split_ratio)
        if training:
            self.items = self.items[:num_train]
            self.labels = self.labels[:num_train]
        else:
            self.items = self.items[num_train:]
            self.labels = self.labels[num_train:]

    def __len__(self):
        return len(self.items * 8) # uid * vid

    def __getitem__(self, idx):

        real_idx = idx // 8
        vid = idx % 8
        
        uid = self.items[real_idx]
        label = self.labels[real_idx]

        results = {}

        # load images (one random view)
        # vid = np.random.choice(8, 1, replace=False)[0]
        
        # image_path = os.path.join(self.path, 'imgs', f'Cap3D_imgs_view{vid}', f'{uid}_{vid}.jpeg')
        image_path = os.path.join(self.path, 'imgs', f'Cap3D_imgs_view{vid}', f'{uid}') # latest dataset changes the path...
        
        images = kiui.read_image(image_path, mode='tensor') # [512, 512, 3] in [0, 1]
        images = images.permute(2, 0, 1).unsqueeze(0) # [1, 3, 512, 512]
        
        # resize & normalize dinov2 input image
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0) # [C, H, W]
        images = TF.normalize(images, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        results['images'] = images # [C, H, W]
        results['labels'] = torch.FloatTensor([label]) # [1]

        return results