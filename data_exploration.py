import os
import cv2
import random
import shutil

def extract_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize frame counter
    frame_count = 0
    
    # Read frames until the end of the video
    while True:
        # Read next frame
        ret, frame = cap.read()
        
        # Check if frame was successfully read
        if not ret:
            break
        
        # Write frame to output directory as RGB image
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Increment frame counter
        frame_count += 1
    
    # Release video capture object
    cap.release()

def process_videos(input_folder, output_folder, num_videos=100):
    # Randomly select videos to process
    selected_videos = random.sample(os.listdir(input_folder), min(num_videos, len(os.listdir(input_folder))))
    
    # Loop through selected videos
    for filename in selected_videos:
        # Check if the file is a video file
        if filename.endswith((".mp4", ".avi", ".mov")):
            # Construct full path to input video file
            video_path = os.path.join(input_folder, filename)
            
            # Create output directory for this video
            video_output_dir = os.path.join(output_folder, os.path.splitext(filename)[0])
            print(video_output_dir)
            
            # Extract frames from video
            extract_frames(video_path, video_output_dir)

if __name__ == "__main__":
    # Example usage
    videos_folder = "C:/Users/isha0/spotting-toxic-clouds/videos"
    output_folder = "C:/Users/isha0/spotting-toxic-clouds/rgb_frames"
    num_videos_to_select = 100

    # Select random videos and process them
    process_videos(videos_folder, output_folder, num_videos=num_videos_to_select)
