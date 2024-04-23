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


class SimCLRVideo(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=1000):
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

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50)
        return [optimizer], [lr_scheduler]

    # def configure_callbacks(self):
    #     checkpoint = ModelCheckpoint(monitor="train_loss")
    #     return [checkpoint]

    def forward(self, x):
        # Forward pass through the base model and projection head
        features = self.model(x)
        projections = self.projection_head(features)
        return projections

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
        self.log(f'{mode}_loss', nll, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return nll

    def training_step(self, batch, batch_idx):
        view1, view2 = batch  # Unpack the batched pairs of views

        # Concatenate the views along the batch dimension to form a larger batch
        # This is useful if you are comparing each view1 with its corresponding view2
        views = torch.cat((view1, view2), dim=0)

        projections = self.forward(views)
        loss = self.info_nce_loss(projections, mode='train')
        return loss
    # def validation_step(self, batch, batch_idx):
    #     views, _ = batch  # Assuming the second part of the batch is labels or something else not used here
    
    #     # If views is not already a tensor, concatenate it. Otherwise, use it directly.
    #     if isinstance(views, (list, tuple)):
    #         views = torch.cat(views, dim=0)
        
    #     projections = self.forward(views)
    #     self.info_nce_loss(projections, mode='val')
    
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

def validate(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in val_loader:
            if inputs.nelement() == 0 or labels.nelement() == 0:
                print("Encountered an empty inputs or labels.")
                continue  
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get the predictions
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    if not all_preds or not all_labels:  # Check if lists are empty
        print("No predictions or labels were collected.")
        return float('nan')
    accuracy = accuracy_score(all_labels, all_preds)
    model.train()  # Set the model back to training mode
    return accuracy

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
    CHECKPOINT_PATH = "./checkpoints"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    torch.set_float32_matmul_precision('medium')

    # if args.model == 'r3d':
    #     # weights = R3D_18_Weights.DEFAULT
    #     # model = r3d_18(weights=weights).to(device).eval()
    #     # model = ModifiedR3D().to(device)
    #     model = SimCLRVideo(hidden_dim=224, lr=1e-3, temperature=0.07, weight_decay=1e-4, max_epochs=50).to(device)
    #     optimizer = optim.Adam(model.parameters(), lr=1e-2)

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
        # val_folder = "./val_labeled_test"
        label_folder = "./metadata_02242020.json"
        dataset = SimCLRDataset(train_folder, transform=train_transforms)
        train_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=7, persistent_workers=True)

        # val_dataset = ValDataset(val_folder, label_folder, transform=train_transforms)
        # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=7)

        pl.seed_everything(42)  # For reproducibility

        # pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
        # if os.path.isfile(pretrained_filename):
        #     print(f'Found pretrained model at {pretrained_filename}, loading...')
        #     # Update to the correct class name and possibly adjust for any required initialization arguments
        #     model = SimCLRVideo.load_from_checkpoint(pretrained_filename)
        #     # optimizer = optim.Adam(model.parameters(), lr=1e-2)

        #     trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt'),
        #     accelerator="gpu" if torch.cuda.is_available() else "cpu",
        #     # devices=1 if torch.cuda.is_available() else None,  # Adjust as per your setup
        #     max_epochs=50,
        #     callbacks=[
        #         ModelCheckpoint(save_weights_only=True, mode='min', monitor='train_loss'),
        #         LearningRateMonitor('epoch')], log_every_n_steps=2)
        # else:

        # Update to the correct class name and pass necessary initialization arguments
        model = SimCLRVideo(hidden_dim=224, lr=1e-3, temperature=0.07, weight_decay=1e-4, max_epochs=1000)
        # optimizer = optim.Adam(model.parameters(), lr=1e-2)
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'Full_SimCLR_test.ckpt'),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # devices=1 if torch.cuda.is_available() else None,  # Adjust as per your setup
        max_epochs=1000,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode='min', monitor='train_loss'),
            LearningRateMonitor('epoch')], log_every_n_steps=2)
        
        trainer.fit(model, train_loader)
        trainer.save_checkpoint(os.path.join(CHECKPOINT_PATH, 'Full_SimCLR_test.ckpt'))
        # Update the checkpoint loading logic if needed
        # model = SimCLRVideo.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # for epoch in range(args.epochs):
        #     for batch_idx, (view1, view2) in enumerate(train_loader):
        #         view1, view2 = view1.to(device), view2.to(device)
        #         repr1, repr2 = model(view1), model(view2)
                
        #         loss = nt_xent_loss(repr1, repr2, temperature=0.5)
                
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
                
        #         if batch_idx % args.pf == 0:
        #             print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
                    
        #     # val_accuracy = validate(model, val_loader, device)
        #     # print(f'Epoch: {epoch}, Validation Accuracy: {val_accuracy}')

        # # Save the model state
        # torch.save(model.state_dict(), './model_state_dict.pth')
        print('Training complete')

    # TODO: Figure out checkpoint, train on bigger batch of data, figure out why larger batch size doesn't work, hyperparameter tuning
    # TODO: Add finetuning
