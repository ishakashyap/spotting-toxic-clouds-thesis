import os
import argparse
import time
import gc
import random
import json
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
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

class SimCLRVideo(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=25, num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'

        # Use a 3D CNN model as base model for video data
        weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=weights)
        # self.model = r3d_18(pretrained=True)  # Pretrained 3D ResNet
        self.model.fc = nn.Identity()  # Remove the final fully connected layer

        # The MLP head for projection
        feature_size = 512  # Known feature size for r3d_18 before the final layer
        self.projection_head = nn.Sequential(
            nn.Linear(feature_size, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        # self.fc = nn.Linear(224, 2)
        # TODO: RUN AGAIN WITH NEW PARAMS

    def forward(self, x):
        # Extract features
        x = self.model(x)

        # Pass through the projection head
        x = self.projection_head(x)

        # Final classification layer
        # x = self.fc(x)
        return x


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def info_nce_loss(self, projections, mode='train'):
        # Calculate cosine similarity
        cos_sim = nn.functional.cosine_similarity(projections[:, None, :], projections[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging
        self.log(f'{mode}_loss', nll, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return nll

    def training_step(self, batch, batch_idx):
        view1, view2 = batch  # Unpack the batched pairs of views

        # Concatenate the views along the batch dimension to form a larger batch
        views = torch.cat((view1, view2), dim=0)

        projections = self.forward(views)
        loss = self.info_nce_loss(projections, mode='train')
        return loss

    
class SimCLRDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.video_files = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        try:
            video_file = self.video_files[idx]
            video_path = os.path.join(self.folder_path, video_file)
            video, _, _ = read_video(video_path, pts_unit='sec')
            video = video.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
            
            # Apply transformations to get two views of the same video
            view1 = self.transform_video(video)
            view2 = self.transform_video(video)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")

        return view1, view2

    def transform_video(self, video):
        try:
            transformed_frames = []
            for frame in video:
                frame = F.to_pil_image(frame)
                if self.transform:
                    frame = self.transform(frame)
                transformed_frames.append(frame)
            video_tensor = torch.stack(transformed_frames)
        except Exception as e:
            print(f"Error transforming video: {e}")

        return video_tensor.permute(1, 0, 2, 3)  # Reshape to (C, T, H, W) for model

def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='simclr', help='simclr/swav')
    # parser.add_argument('--dataset', type=str, default='/.videos', help='path to videos')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, help='log directory')
    # parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run')
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
    CHECKPOINT_PATH = "./checkpoints"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    torch.set_float32_matmul_precision('medium')

    if args.mode == 'train':
         # Try different transformations
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
        
        train_folder = "./full_dataset"
        label_folder = "./metadata_12012023.json"
        dataset = SimCLRDataset(train_folder, transform=train_transforms)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=7, persistent_workers=True)

        early_stop = EarlyStopping(
            monitor='train_loss',
            min_delta=0.0,
            patience=3,
            verbose=True,
            mode='min'
        )

        pl.seed_everything(42)  # For reproducibility

        # pretrained_filename = './checkpoints/Full_SimCLR_test.ckpt/lightning_logs/version_1/checkpoints/epoch=18-step=5700.ckpt' #os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
        # if os.path.isfile(pretrained_filename):
        #     print(f'Found pretrained model at {pretrained_filename}, loading...')
        #     # Update to the correct class name and possibly adjust for any required initialization arguments
        #     model = SimCLRVideo.load_from_checkpoint(pretrained_filename)
        #     # optimizer = optim.Adam(model.parameters(), lr=1e-2)

        # Update to the correct class name and pass necessary initialization arguments
        # else:
        model = SimCLRVideo(hidden_dim=224, lr=1e-3, temperature=0.07, weight_decay=1e-4, max_epochs=25)
        # optimizer = optim.Adam(model.parameters(), lr=1e-2)
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'final_data.pth'),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # devices=1 if torch.cuda.is_available() else None,  # Adjust as per your setup
        max_epochs=25,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode='min', monitor='train_loss'),
            LearningRateMonitor('epoch')])
        
        trainer.fit(model, train_loader)
        optimizer = trainer.optimizers[0]

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'final_data.pth')

        print('Training complete')