import os
import json
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.io import read_video
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from imblearn.over_sampling import SMOTE
from torchvision.transforms import functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import classification_report, f1_score, confusion_matrix

class VideoDataset(Dataset):
    def __init__(self, folder_path, labels_json_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        
        # Load video filenames
        self.video_files = os.listdir(folder_path)
        
        # Load labels from JSON file
        self.labels = self._load_labels(labels_json_path)
        self.video_files = self.validate_videos_and_labels()

        self.targets = [self.labels[video_file] for video_file in self.video_files]
        self.targets = torch.tensor(self.targets)

    def _load_labels(self, labels_json_path):
        with open(labels_json_path, 'r') as f:
            labels_json = json.load(f)
        # Append the .mp4 extension to the filenames
        labels = {item['file_name'] + '.mp4': item['label'] 
                    for item in labels_json 
                    if item['label'] is not None and item['label'] != -1}
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
        # label = adjust_labels(label)
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
    
    
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cm_filename='conf_matrix.png', cr_filename='clf_report.txt'):
    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title(title)
    # plt.savefig(cm_filename)
    # plt.show()
    # plt.close()

    report = classification_report(y_true, y_pred, target_names=classes)
    with open(cr_filename, 'w') as f:
        f.write(report)

def get_oversampled_loader(dataset):
    targets = []
    for _, label in dataset:
        if label is not None:
            targets.append(label.item())
            
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in targets])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

    # sampled_targets = [targets[i] for i in list(sampler)]
    # print_class_distribution(sampled_targets, "Class Distribution After Sampling")

    return sampler

def get_smote_dataset(dataset):
    data, labels = [], []
    for view, label in dataset:
        data.append(view.flatten())  # Flatten for SMOTE compatibility
        labels.append(label)
    
    smote = SMOTE()
    data_resampled, labels_resampled = smote.fit_resample(data, labels)
    
    resampled_dataset = [(data_resampled[i].reshape(view.shape), labels_resampled[i]) for i in range(len(data_resampled))]
    return resampled_dataset

def train(train_loader, val_loader, test_loader, model, optimizer, criterion, num_epochs, scheduler):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []

        for views, labels in train_loader:
            if views is None or labels is None:
                continue

            views, labels = views.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(views)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * views.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = train_loss / len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, LR: {current_lr}")
        print('Training Classification Report:')
        print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
        # plot_confusion_matrix(all_labels, all_preds, classes=['Class 0', 'Class 1'], title=f'Training Confusion Matrix Epoch {epoch+1}', cm_filename=f'training_confusion_matrix_epoch_{epoch+1}.png', cr_filename=f'training_classification_report_epoch_{epoch+1}.txt')

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for views, labels in val_loader:
                if views is None or labels is None:
                    continue

                views, labels = views.cuda(), labels.cuda()
                outputs = model(views)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * views.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(epoch_val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}')
        print('Validation Classification Report:')
        print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
        # plot_confusion_matrix(all_labels, all_preds, classes=['Class 0', 'Class 1'], title=f'Validation Confusion Matrix Epoch {epoch+1}', cm_filename=f'validation_confusion_matrix_epoch_{epoch+1}.png', cr_filename=f'validation_classification_report_epoch_{epoch+1}.txt')

        test(test_loader=test_loader, model=model, criterion=criterion)

def test(test_loader, model, criterion):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for views, labels in test_loader:
            if views is None or labels is None:
                continue

            views, labels = views.cuda(), labels.cuda()
            outputs = model(views)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * views.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {epoch_loss:.4f}')
    print('Test Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))

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
    
    train_folder = "./train_set"
    train_label_folder = "./split/metadata_train_split_by_date.json"

    val_folder = "./validation_set"
    val_label_folder = "./split/metadata_validation_split_by_date.json"

    test_folder = "./test_set"
    test_label_folder = "./split/metadata_test_split_by_date.json"

    train_dataset = VideoDataset(train_folder, train_label_folder, transform=train_transforms)
    val_dataset = VideoDataset(val_folder, val_label_folder, transform=train_transforms)
    test_dataset = VideoDataset(test_folder, test_label_folder, transform=train_transforms)

    # train_sampler = get_oversampled_loader(train_dataset)
    # val_sampler = get_oversampled_loader(val_dataset)
    # test_sampler = get_oversampled_loader(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    
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

    # train_labels = extract_labels_from_dataset(train_dataset)
    # class_weights = calculate_class_weights(train_labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(self_supervised_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min')
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # if 'optimizer_state_dict' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Train and evaluate the model
    train(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, model=self_supervised_model, optimizer=optimizer, criterion=criterion, num_epochs=6, scheduler=scheduler) # , patience=5, min_delta=0.001

    # Save the trained model
    # torch.save(self_supervised_model.state_dict(), 'linear_eval_model.pth')

if __name__ == "__main__":
    main()