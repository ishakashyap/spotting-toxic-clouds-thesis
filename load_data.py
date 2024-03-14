import random
import os
import argparse

def move_videos(video_list, destination_folder, source_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for video in video_list:
        source_path = os.path.join(source_folder, video)
        destination_path = os.path.join(destination_folder, video)
        os.rename(source_path, destination_path)

def main(source_folder, train_folder, test_folder, val_folder, split_ratio):
    # for folder in [source_folder, train_folder, test_folder, val_folder]:
    #     if not os.path.exists(folder):
    #         print(f"Error: Folder {folder} does not exist.")
    #         return
    
    video_files = os.listdir(source_folder)
    random.shuffle(video_files)

    train_split = int(split_ratio[0] * len(video_files))
    test_split = int(split_ratio[1] * len(video_files))
    val_split = int(split_ratio[2] * len(video_files))

    train_videos = video_files[:train_split]
    test_videos = video_files[train_split:train_split+test_split]
    val_videos = video_files[train_split+test_split:]

    move_videos(train_videos, train_folder, source_folder)
    move_videos(test_videos, test_folder, source_folder)
    move_videos(val_videos, val_folder, source_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and move videos into train, test, and validation sets.")
    parser.add_argument("source_folder", help="Path to the source folder containing videos.")
    parser.add_argument("train_folder", help="Path to the folder where train videos will be moved.")
    parser.add_argument("test_folder", help="Path to the folder where test videos will be moved.")
    parser.add_argument("val_folder", help="Path to the folder where validation videos will be moved.")
    parser.add_argument("split_ratio", nargs=3, type=float, help="Split ratio for train, test, and validation sets.")

    args = parser.parse_args()
    main(args.source_folder, args.train_folder, args.test_folder, args.val_folder, args.split_ratio)


