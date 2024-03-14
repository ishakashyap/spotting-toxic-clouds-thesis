import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomApply, RandomRotation, GaussianBlur, RandomGrayscale, ColorJitter
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image
import os

# Initialize the model and set to evaluation mode
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights).eval()

# Augmentation and transformation of the data
transform = Compose([
    Resize(256), 
    RandomApply([GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
    RandomApply([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
    RandomGrayscale(p=0.2),
    RandomHorizontalFlip(), 
    RandomRotation(degrees=15),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# TODO: Look at augmented data

# Loop over frames for every video, convert to RGB, stack frames and unsqueeze
def preprocess_video_frames(frames_folder):
    frames = []
    print("Reading Images... \n")
    for frame_file in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame_file)
        frame = Image.open(frame_path).convert("RGB")
        frames.append(transform(frame))
    
    if len(frames) > 0:
        print("Images have been read... \n")
        frames_tensor = torch.stack(frames)  # Shape: [depth, 3, 224, 224]
        frames_tensor = frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # Shape: [1, 3, depth, 224, 224]
        return frames_tensor
    else:
        return None

def extract_features(input_tensor):
    with torch.no_grad():
        features = model(input_tensor)
    return features

frames_folder = "C:/Users/isha0/spotting-toxic-clouds/rgb_frames"
output_folder = "C:/Users/isha0/spotting-toxic-clouds/features"

i = 0

for video_folder in os.listdir(frames_folder):
    video_path = os.path.join(frames_folder, video_folder)
    
    if os.path.isdir(video_path):
        i += 1
        video_tensor = preprocess_video_frames(video_path)
        if video_tensor is not None:
            print(f"Extracting features for video: {i} \n")
            features = extract_features(video_tensor)
            torch.save(features, os.path.join(output_folder, video_folder + ".pt"))
        else:
            print(f"No frames found for video {video_folder}")

print('done')