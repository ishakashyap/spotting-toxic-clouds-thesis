import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_video
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import classification_report, f1_score

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

class VideoDataset(Dataset):
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
        # Keep only videos for which we have a valid label and can be loaded
        valid_videos = []
        for video_file in os.listdir(self.folder_path):
            if video_file in self.labels:  # Checks if video has a valid label
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
        
        # Get the label for the current video
        label = self.labels.get(video_file, None)

        if label is None:
            print(f"Failed to load label: {video_path}")
            return None, None
        
        label = torch.tensor(label)
        label = adjust_labels(label)
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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')
            print(f'{phase} classification Report:')
            print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))


def calculate_class_weights(dataset):
    class_counts = np.bincount(dataset.labels)
    weights = 1. / class_counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float)

def main():

    train_transforms = transforms.Compose([
        transforms.Resize(224), 
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(degrees=15),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_folder = "./train_baseline"
    train_label_folder = "./split/metadata_train_split_by_date.json"

    val_folder = "./validation_baseline"
    val_label_folder = "./split/metadata_validation_split_by_date.json"

    test_folder = "./test_baseline"
    test_label_folder = "./split/metadata_test_split_by_date.json"

    train_dataset = VideoDataset(train_folder, train_label_folder, transform=train_transforms)
    val_dataset = VideoDataset(val_folder, val_label_folder, transform=train_transforms)
    test_dataset = VideoDataset(test_folder, test_label_folder, transform=train_transforms)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True),
        # 'test': DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained ResNet model and modify the final layer
    model_path = "SimCLR_full_data.pth"
    weights = R3D_18_Weights.DEFAULT
    self_supervised_model  = r3d_18(weights=weights)
    self_supervised_model.fc = nn.Identity()

    checkpoint = torch.load(model_path, map_location='cpu')
    self_supervised_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Freeze all layers of the pre-trained model
    for param in self_supervised_model.parameters():
        param.requires_grad = False

    # Add a linear layer on top for the classification task
    num_ftrs = 512 #self_supervised_model.fc.in_features
    self_supervised_model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification

    self_supervised_model = self_supervised_model.to(device)

    class_weights = calculate_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
    optimizer = optim.SGD(self_supervised_model.fc.parameters(), lr=0.001, momentum=0.9)

    # if 'optimizer_state_dict' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Train and evaluate the model
    model = train_model(self_supervised_model, dataloaders, criterion, optimizer, num_epochs=2)
    # test_model(model, dataloaders['test'], criterion)

    # Save the trained model
    torch.save(model.state_dict(), 'linear_eval_model.pth')

if __name__ == "__main__":
    main()