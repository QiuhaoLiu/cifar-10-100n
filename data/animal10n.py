# animal10n.py

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob

class Animal10N(Dataset):
    def __init__(self, 
                 root: str = '~/data/animal10n',   # 图片根目录
                 train: bool = True,
                 transform=None,
                 target_transform=None,
                 label_pt_path: str = 'mixup-cifar10-main/data/ANIMAL-10N.pt'  # 标签文件
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.nb_classes = 10

        # 1) 加载标签文件
        label_data = torch.load(label_pt_path)

        if self.train:
            self.labels = label_data['train_labels']
            self.image_dir = os.path.join(self.root, 'train')
        else:
            self.labels = label_data['test_labels']
            self.image_dir = os.path.join(self.root, 'test')

        # 2) 获取所有图片路径（假设都是 png）
        self.image_paths = sorted(glob(os.path.join(self.image_dir, '*.png')))

        # 3) 数据长度（验证图片数和标签是否匹配）
        assert len(self.image_paths) == len(self.labels), "图片数量和标签数量不一致！"
        self.length = len(self.image_paths)

        print(f'Loaded Animal10N {"train" if self.train else "test"} dataset:')
        print(f'- Number of images: {self.length}')
        print(f'- Image directory: {self.image_dir}')
        print(f'- Labels loaded from: {label_pt_path}')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 1) 加载图片路径
        img_path = self.image_paths[index]

        # 2) 打开图片（转为 PIL）
        image_pil = Image.open(img_path).convert('RGB')  # 保证是 RGB 格式

        # 3) 获取标签
        label = int(self.labels[index])

        # 4) 做 transforms
        if self.transform is not None:
            image_pil = self.transform(image_pil)

        if self.target_transform is not None:
            label = self.target_transform(label)

        # 5) 返回 image, label, index
        
        return image_pil, label, index
