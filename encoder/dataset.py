import os
import pickle
import random
from functools import lru_cache
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import torch
from tqdm import trange
import pytorch_lightning as pl
import glob
from tqdm import trange
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights


def getfromidnex(labeldata, data):
    mask = (labeldata['dataname'] == data[0].split('-')[0]) & (labeldata['group'] == data[0].split('-')[1])
    result = labeldata[mask]

    for i in trange(1, len(data)):
        mask = (labeldata['dataname'] == data[i].split('-')[0]) & (labeldata['group'] == data[i].split('-')[1])
        result = pd.concat([result, labeldata[mask]])
    return result


@lru_cache(maxsize=None)
def load_image(path):
    return Image.open(path).convert("RGB")


def load_frames(image_paths):
    frames = []
    for image in image_paths:
        frames.append(load_image(image))
    return frames


class VideoDataset(Dataset):
    def __init__(self, args, labeldata, data, transform=None, stage='test'):
        self.data_dir = args.data_dir
        self.num_frames = args.num_frames
        self.labeldata = labeldata
        self.transforms = transform
        self.result = []
        self.names = []
        self.path = []
        result = getfromidnex(labeldata, data).values
        for i in trange(result.shape[0]):
            filename = os.path.splitext(result[i][3])[0]
            samplename = filename
            if samplename in self.names:
                continue
            self.names.append(samplename)
            file_list = sorted(
                glob.glob(os.path.join(args.data_dir,  '*', args.type, samplename + '*',
                                       '*.jpg')))
            file_list = file_list[::args.frame_interval]

            self.result = self.result + file_list

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        image_paths = [self.result[idx]]

        frames = load_frames(image_paths)

        if self.transforms:
            frames = self.transforms(frames[0])
        score = self.labeldata[
            self.labeldata['filename'] == image_paths[0].split('/')[-2].replace('_aligned', '') + '.mp4'].values[0][2]
        return frames.float(), torch.tensor(score).float(), image_paths[0]


class VideoRegressionDataModule(pl.LightningDataModule):
    # def __init__(self, data_dir, label_file, train_data, val_data, test_data, num_frames=16, batch_size=32,
    #              num_workers=4, frame_interval=2, type='image', cache_path=None):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.num_frames = args.num_frames
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.frame_interval = args.frame_interval

        self.type = args.type
        self.save_hyperparameters(args)
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(112, antialias=False),
            transforms.RandomHorizontalFlip(p=0.9),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(112, antialias=False),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 读取标签对应文件
        self.labeldata = pd.read_csv(args.label_file)

        self.train_dataset = VideoDataset(args, self.labeldata, args.train_data,
                                          transform=train_transforms, stage='train')
        self.val_dataset = VideoDataset(args, self.labeldata, args.val_data,
                                        transform=val_transform, )
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers)
        return test_loader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
