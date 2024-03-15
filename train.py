import os
import argparse
import time
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import torch.optim as optim
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomApply, RandomRotation, GaussianBlur, RandomGrayscale, ColorJitter
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image

class LoadDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.video_files = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.folder_path, video_file)
        video, audio, info = read_video(video_path, pts_unit='sec')

        if self.transform:
            video = self.transform(video)

        return video

def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='c3d', help='c3d/r3d/r21d')
    # parser.add_argument('--dataset', type=str, default='/.videos', help='path to videos')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, help='log directory')
    # parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--epochs', type=int, default=150, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == 'r3d':
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights).to(device).eval()

    if args.mode == 'train':
        train_transforms = Compose([
            Resize(64), 
            RandomApply([GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
            RandomApply([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            RandomGrayscale(p=0.2),
            RandomHorizontalFlip(), 
            RandomRotation(degrees=15),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_folder = "./train"
        
        train = LoadDataset(train_folder, transform=train_transforms)
        print(len(train))
        train_dataloader = DataLoader(train, batch_size=1, shuffle=True, num_workers=args.workers)

        for epoch in range(args.start_epoch, args.epochs + 1):
            for i, data in enumerate(train_dataloader, 0):
                # Move data to the correct device
                data = data.to(device)

                # Example forward pass (model training steps should go here)
                # output = model(data)
                # loss = loss_function(output, labels)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # Clear memory
                del data  # Delete the variables to free up memory
                torch.cuda.empty_cache()  # Clear unused memory

                if i % args.pf == 0:
                    print(f'Epoch: {epoch}, Batch: {i}')
                    # Optionally monitor memory usage at intervals
                    # print(torch.cuda.memory_summary())

            gc.collect()  # Explicitly call garbage collection at the end of an epoch (optional)

        print('Training complete')