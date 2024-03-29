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
from torchvision.transforms import functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
    
class ModifiedR3D(nn.Module):
    def __init__(self):
        super(ModifiedR3D, self).__init__()
        # self.weights = R3D_18_Weights.DEFAULT
        # self.model = r3d_18(weights=self.weights).to(device).eval()
        self.model = r3d_18(pretrained=False, progress=True)
        self.model.fc = nn.Identity()  # Remove the final fully connected layer
        self.projection_head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)

    def forward(self, x):
        features = self.model(x)
        projections = self.projection_head(features)
        return projections

class SimCLRDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.video_files = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.folder_path, video_file)
        video, _, _ = read_video(video_path, pts_unit='sec')
        video = video.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
        
        # Apply transformations to get two views of the same video
        view1 = self.transform_video(video)
        view2 = self.transform_video(video)
        return view1, view2

    def transform_video(self, video):
        transformed_frames = []
        for frame in video:
            frame = F.to_pil_image(frame)
            if self.transform:
                frame = self.transform(frame)
            transformed_frames.append(frame)
        video_tensor = torch.stack(transformed_frames)
        return video_tensor.permute(1, 0, 2, 3)  # Reshape to (C, T, H, W) for model

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Calculate the normalized temperature-scaled cross entropy loss.
    Assumes z_i and z_j are normalized embeddings of shape (batch_size, feature_dim).
    """
    cos_sim = torch.matmul(z_i, z_j.T) / temperature
    # Assuming z_i and z_j are L2-normalized, cos_sim computes the cosine similarity
    labels = torch.arange(0, z_i.size(0)).to(z_i.device)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)
    return loss

# class LoadDataset(Dataset):
#     def __init__(self, folder_path, transform=None):
#         self.folder_path = folder_path
#         self.video_files = os.listdir(folder_path)
#         self.transform = transform

#     def __len__(self):
#         return len(self.video_files)

#     def __getitem__(self, idx):
#         video_file = self.video_files[idx]
#         video_path = os.path.join(self.folder_path, video_file)
#         video, audio, info = read_video(video_path, pts_unit='sec')

#         video = video.permute(0, 3, 1, 2)
#         print(video.size)
    
#         transformed_frames = []
#         for frame in video:
#             # frame = video[t]  # frame is now in shape (C, H, W)
#             # frame = frame.float() / 255.0  # Normalize to [0, 1]
#             frame = F.to_pil_image(frame)  # Convert to PIL Image

#             if self.transform is not None:
#                 frame = self.transform(frame)  # Apply transformations

#             # frame = F.to_tensor(frame)  # Convert back to tensor
#             transformed_frames.append(frame)

#         video = torch.stack(transformed_frames)

#         print(video.size())

#         video = video.permute(1, 0, 2, 3) 
#         print(video.size())
#         return video

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
        # weights = R3D_18_Weights.DEFAULT
        # model = r3d_18(weights=weights).to(device).eval()
        model = ModifiedR3D().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

    if args.mode == 'train':
        train_transforms = Compose([
            Resize(224), 
            RandomApply([GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
            RandomApply([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            RandomGrayscale(p=0.2),
            RandomHorizontalFlip(), 
            RandomRotation(degrees=15),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_folder = "./train_full"
        dataset = SimCLRDataset(train_folder, transform=train_transforms)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

        for epoch in range(args.epochs):
            for batch_idx, (view1, view2) in enumerate(train_loader):
                view1, view2 = view1.to(device), view2.to(device)
                repr1, repr2 = model(view1), model(view2)
                # Implement or use an existing contrastive loss function
                loss = nt_xent_loss(repr1, repr2, temperature=0.5)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx % args.pf == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

        # train = LoadDataset(train_folder, transform=train_transforms)
        # print(len(train))
        # train_dataloader = DataLoader(train, batch_size=1, shuffle=True, num_workers=args.workers)
        
        # criterion = nn.MSELoss()
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

        # for epoch in range(args.start_epoch, args.epochs + 1):
        #     running_loss = 0.0
        #     for i, inputs in enumerate(train_dataloader, 1):
        #         inputs = inputs.to(device)

        #         optimizer.zero_grad()
        #         print("Input shape to model:", inputs.shape)
        #         outputs = model(inputs)
        #         # For self-supervised, you might compare outputs to inputs or use another form of self-supervision
        #         loss = criterion(outputs, inputs)  # This is an example and likely needs adjustment
        #         loss.backward()
        #         optimizer.step()

        #         running_loss += loss.item()

        #         if i % args.pf == 0:
        #             print(f'Epoch: {epoch}, Batch: {i}, Loss: {running_loss / i:.4f}')

        #     print(f'Epoch {epoch} complete, Average Loss: {running_loss / len(train_dataloader):.4f}')
        #     gc.collect()

        print('Training complete')