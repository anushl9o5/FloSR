import cv2
import os
import argparse
import csv
import numpy as np

from tqdm import tqdm

def create_video_from_csv(base_dir, csv_file, output_file, fps=30, target_width=2048, target_height=512):
    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Open the CSV file
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        # Read each row and create a list of image paths
        image_paths_list = [[row[-2], row[-1]] for row in reader]

    # Ensure that each row contains two image paths
        
    for row in image_paths_list:
        if len(row) != 2:
            raise ValueError("Each row in the CSV file must contain two image paths separated by a comma.")

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))

    # Iterate through the image paths list
    for image_path1, image_path2 in tqdm(image_paths_list):
        # Read images
        img1 = cv2.imread(os.path.join(base_dir, image_path1))
        img2 = cv2.imread(os.path.join(base_dir, image_path2))

        # Resize images
        img1 = cv2.resize(img1, (target_width // 2, target_height))
        img2 = cv2.resize(img2, (target_width // 2, target_height))

        # Create a blank image with white background
        blank_image = 255 * np.ones((target_height, target_width, 3), np.uint8)

        # Place images side by side horizontally
        blank_image[:, :target_width//2] = img1
        blank_image[:, target_width//2:] = img2

        # Write the frame to the video
        video.write(blank_image)

    # Release the VideoWriter object
    video.release()

def main():
    parser = argparse.ArgumentParser(description='Create a video from images listed in a CSV file.')
    parser.add_argument('--data_dir', required=True, help='Path to dataset base directory')
    parser.add_argument('--csv_file', required=True, help='Path to the CSV file containing image paths.')
    parser.add_argument('--output_file', required=True, help='Path to the output video file.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the video.')
    parser.add_argument('--width', type=int, default=2048, help='Target width for resizing images.')
    parser.add_argument('--height', type=int, default=512, help='Target height for resizing images.')

    args = parser.parse_args()

    create_video_from_csv(args.data_dir, args.csv_file, args.output_file, args.fps, args.width, args.height)

if __name__ == "__main__":
    main()