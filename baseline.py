import os
import argparse
import time
import gc
import random
import json
import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch.cuda.amp import GradScaler, autocast
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
from train import SimCLRVideo
import logging

logging.basicConfig(level=logging.INFO)


def adjust_labels(y):
    # Detach y to ensure no gradients are backpropagated through the label adjustment
    y = y.detach()

    # Initialize all labels to a default value (e.g., -1 for filtering or 0 if using binary classification)
    y_adjusted = torch.full_like(y, -1)

    # Mapping specific labels
    y_adjusted[y == 47] = 1  # Gold standard positive
    y_adjusted[y == 32] = 0  # Gold standard negative
    y_adjusted[y == 23] = 1  # Strong positive
    y_adjusted[y == 16] = 0  # Strong negative
    y_adjusted[y == 19] = 1  # Weak positive
    y_adjusted[y == 20] = 0  # Weak negative

    return y_adjusted

class SimCLR_eval(pl.LightningModule):
    def __init__(self, lr, num_classes=2):
        super().__init__()
        self.lr = lr

        weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(top_k=1, task='binary')
        self.top5_accuracy = torchmetrics.Accuracy(top_k=5, task='binary')
        self.epoch_accuracies = []
        self.scaler = GradScaler()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        try:
            x, y = batch
            y = adjust_labels(y)  # Make sure this function does not retain any graph

            with torch.no_grad():
                logits = self.forward(x)  # Get model predictions
            
            loss = self.loss(logits, y)

            self.optimizer.zero_grad()

            # Perform backward pass and scale loss under autocast
            loss.requires_grad = True
            loss.backward()
            self.optimizer.step()


            # with torch.no_grad():
            _, preds = torch.max(logits, dim=1)
            acc = self.accuracy(preds, y)
            top5_acc = self.top5_accuracy(preds, y)

            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train_top5_acc', top5_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            avg_acc = self.accuracy.update(preds, y)
            avg_top5_acc = self.top5_accuracy.update(preds, y)

            self.accuracy.reset()
            self.top5_accuracy.reset()

        except Exception as e:
            logging.error('Error in training_step: ', e)

        return {'loss': loss, 'train_acc': acc, 'train_top5_acc': top5_acc}
    
    # def on_train_epoch_end(self):
    #     avg_acc = self.accuracy.compute()
    #     avg_top5_acc = self.top5_accuracy.compute()
    #     self.epoch_accuracies.append(avg_acc.item())  # Store the average Top-1 accuracy of the epoch
    #     self.log('avg_train_acc', avg_acc, sync_dist=True)
    #     self.log('avg_train_top5_acc', avg_top5_acc, sync_dist=True)
    #     self.accuracy.reset()
    #     self.top5_accuracy.reset()
    
    # def on_train_end(self):
    #     overall_avg_accuracy = np.mean(self.epoch_accuracies)
    #     print(f'Overall Average Top-1 Accuracy across all epochs: {overall_avg_accuracy}')

    def validation_step(self, batch, batch_idx):
    #    with torch.no_grad():
        try:
            x, y = batch
            y = adjust_labels(y)
            logits = self(x)
            loss = self.loss(logits, y)
            _, preds = torch.max(logits, dim=1)
            acc = self.accuracy(preds, y)
            top5_acc = self.top5_accuracy(preds, y)

            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_top5_acc', top5_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            avg_acc = self.accuracy.update(preds, y)
            avg_top5_acc = self.top5_accuracy.update(preds, y)
        except Exception as e:
            logging.error('Error in validation step: ', e)
        return {'loss': loss, 'val_acc': acc, 'val_top5_acc': top5_acc}
    
    # def on_validation_epoch_end(self):
    #     avg_acc = self.accuracy.compute()
    #     avg_top5_acc = self.top5_accuracy.compute()
    #     self.epoch_accuracies.append(avg_acc.item())  # Store the average Top-1 accuracy of the epoch
    #     self.log('avg_val_acc', avg_acc, sync_dist=True)
    #     self.log('avg_val_top5_acc', avg_top5_acc, sync_dist=True)
    #     self.accuracy.reset()
    #     self.top5_accuracy.reset()
    
    # def on_validation_end(self):
    #     overall_avg_accuracy = np.mean(self.epoch_accuracies)
    #     print(f'Overall Average Top-1 Validation Accuracy across all epochs: {overall_avg_accuracy}')

    def on_epoch_end(self):
        # Get the average accuracy from the current epoch for both training and validation
        train_acc = self.trainer.callback_metrics.get('train_acc')
        val_acc = self.trainer.callback_metrics.get('val_acc')
        
        # Create or open the log file and append the current epoch's accuracies
        with open('epoch_accuracy_log.txt', 'a') as f:
            f.write(f'Epoch {self.current_epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}\n')

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer

class LabeledDataset(Dataset):
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
                    if item['label_state_admin'] is not None and item['label_state_admin'] != -1}
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
    parser = argparse.ArgumentParser(description='Baseline')
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
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True

    early_stop = EarlyStopping(
            monitor='train_loss',
            min_delta=0.0,
            patience=3,
            verbose=True,
            mode='min'
    )

    # Currenly commented out transformation in dataset loader
    train_transforms = Compose([
        Resize(224), 
        RandomApply([GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
        RandomApply([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        RandomGrayscale(p=0.2),
        RandomHorizontalFlip(), 
        # RandomRotation(degrees=15),
        # CenterCrop(224),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_folder = "./train_baseline"
    train_label_folder = "./split/metadata_train_split_by_date.json"

    val_folder = "./validation_baseline"
    val_label_folder = "./split/metadata_validation_split_by_date.json"
    
    test_folder = "./test_baseline"
    test_label_folder = "./split/metadata_test_split_by_date.json"

    train_dataset = LabeledDataset(train_folder, train_label_folder, transform=train_transforms)
    val_dataset = LabeledDataset(val_folder, val_label_folder, transform=train_transforms)
    # test_dataset = LabeledDataset(test_folder, test_label_folder, transform=train_transforms)

    # TODO Check rise paper for split technique, split by cam view, check split_metadata.json
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoader for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)


    pl.seed_everything(42)  # For reproducibility

    # model = SimCLRVideoLinearEval(lr=1e-3, hidden_dim=224, weight_decay=5e-4, num_classes=2)
    model = SimCLR_eval(lr=1e-3, num_classes=2)

    # Update to the correct class name and possibly adjust for any required initialization arguments
    # model_state_dict = checkpoint['state_dict']
    # optimizer_state_dict = checkpoint.get('optimizer_state', None)
    # sim_model = SimCLRVideo.load_from_checkpoint(pretrained_filename)
    # backbone_model = sim_model.model
    # model = SimCLR_eval(lr=1e-3, model=None, fine_tune=True, linear_eval=False, accumulation_steps=20)
    
    # checkpoint = torch.load(pretrained_filename, map_location='cpu')
    # # print(checkpoint.keys())
    # for key in list(checkpoint.keys()):
    #     print(key)
    #     if 'model.' in key:
    #         checkpoint[key.replace('model.', '')] = checkpoint[key]
    #         del checkpoint[key]
    # model.load_state_dict(checkpoint)

    # fine_tuning_model = SimCLR_eval(lr=1e-3, model=backbone_model, fine_tune=True, linear_eval=False, accumulation_steps=20)
    # optimizer = optim.Adam(model.parameters(), lr=1e-2)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[ModelCheckpoint(dirpath='./checkpoints/', monitor='train_acc', mode='max'), early_stop], log_every_n_steps=2
    )

    # Start the training and validation process
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)