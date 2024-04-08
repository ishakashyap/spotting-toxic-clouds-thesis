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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class SimCLR_eval(pl.LightningModule):
    def __init__(self, lr, model=None, linear_eval=False, fine_tune=False):
        super().__init__()
        self.lr = lr
        self.linear_eval = linear_eval
        self.fine_tune = fine_tune
        
        # Ensure the base model is in training mode if we're fine-tuning
        if self.fine_tune:
            model.train()
        elif self.linear_eval:
            model.eval()  # Only in linear_eval mode, we keep the base model in eval mode

        self.mlp = nn.Sequential(
            nn.Linear(512, 10),  # Adjust the in_features to match your model's output
        )

        # Incorporate the base model with the newly added MLP for classification
        self.model = model
        self.classifier = self.mlp
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):
        features = self.model(X)  # Get features from the base model
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
       x, y = batch
       z = self.forward(x)
       loss = self.loss(z, y)
       self.log('Cross Entropy loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

       predicted = z.argmax(1)
       acc = (predicted == y).sum().item() / y.size(0)
       self.log('Train Acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

       return loss

    def validation_step(self, batch, batch_idx):
       x, y = batch
       z = self.forward(x)
       loss = self.loss(z, y)
       self.log('Val CE loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

       predicted = z.argmax(1)
       acc = (predicted == y).sum().item() / y.size(0)
       self.log('Val Accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

       return loss

    def configure_optimizers(self):
    # Different learning rate for fine-tuning pre-trained layers
        base_lr = self.lr / 10
        classifier_lr = self.lr

        optimizer = optim.SGD([
            {'params': self.model.parameters(), 'lr': base_lr},
            {'params': self.classifier.parameters(), 'lr': classifier_lr},
        ], lr=self.lr, momentum=0.9)

        return optimizer

class ValDataset(Dataset):
    def __init__(self, folder_path, labels_json_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        
        # Load video filenames
        self.video_files = os.listdir(folder_path)
        
        # Load labels from JSON file
        self.labels = self._load_labels(labels_json_path)
        self.video_files = self.validate_videos_and_labels()

    def _load_labels(self, labels_json_path):
        with open(labels_json_path, 'r') as f:
            labels_json = json.load(f)
        # Append the .mp4 extension to the filenames
        labels = {item['file_name'] + '.mp4': item['label_state_admin'] 
                    for item in labels_json 
                    if item['label_state_admin'] is not None}
        return labels
    
    def validate_videos_and_labels(self):
        # Keep only videos for which we have a non-None label and can be loaded
        valid_videos = []
        for video_file in os.listdir(self.folder_path):
            if video_file in self.labels:  # Checks if video has a non-None label
                video_path = os.path.join(self.folder_path, video_file)
                try:
                    video, _, _ = read_video(video_path, pts_unit='sec')
                    if video.nelement() > 0:  # Checks if video is loaded properly
                        valid_videos.append(video_file)
                except Exception as e:
                    print(f"Error loading video {video_file}: {e}")
        return valid_videos

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.folder_path, video_file)
        
        # Load the video
        video, _, _ = read_video(video_path, pts_unit='sec')

        if video.nelement() == 0:  # or any other condition indicating failure
            # Handle error: log, raise an exception, or return a default value
            print(f"Failed to load video: {video_path}")
            return None, None
    
        video = video.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
        
        # Apply transformation to get a single view of the videos
        view = self.transform_video(video)
        
        # Get the label for the current video, default to None if not found
        label = self.labels.get(video_file, None)

        if label is None:
            print(f"Failed to load label: {video_path}")
            return None, None
        
        return view, label

    def transform_video(self, video):
        transformed_frames = []
        for frame in video:
            frame = F.to_pil_image(frame)
            if self.transform:
                frame = self.transform(frame)
            transformed_frames.append(frame)
        video_tensor = torch.stack(transformed_frames)
        return video_tensor.permute(1, 0, 2, 3) 

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
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
    CHECKPOINT_PATH = "./checkpoints"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    torch.set_float32_matmul_precision('medium')

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
    
    val_folder = "./val_labeled_test"
    label_folder = "./metadata_02242020.json"

    val_dataset = ValDataset(val_folder, label_folder, transform=train_transforms)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=7)

    pl.seed_everything(42)  # For reproducibility

    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    print(f'Found pretrained model at {pretrained_filename}, loading...')
    # Update to the correct class name and possibly adjust for any required initialization arguments
    model = SimCLR_eval.load_from_checkpoint(pretrained_filename)
    fine_tuning_model = SimCLR_eval(lr=1e-3, model=model, fine_tune=True)
    # optimizer = optim.Adam(model.parameters(), lr=1e-2)

    trainer = pl.Trainer(
        max_epochs=50,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[ModelCheckpoint(dirpath='./checkpoints/', monitor='Train Acc', mode='max')],
    )

    # Start the training and validation process
    trainer.fit(fine_tuning_model, train_dataloaders=val_loader)