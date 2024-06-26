import cv2
import os
import argparse
from tqdm import tqdm

def create_video(input_folder, output_file, fps=30, target_width=1024, target_height=512):
    # Get the list of PNG files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    # Sort the image files based on their names
    image_files.sort()

    # Read the dimensions of the first image to set video dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, layers = first_image.shape

    # Resize dimensions
    resize_dimensions = (target_width, target_height)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, resize_dimensions)

    # Iterate through the sorted image files, resize them, and add to the video
    for image_file in tqdm(sorted(image_files)):
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        resized_frame = cv2.resize(frame, resize_dimensions)
        video.write(resized_frame)

    # Release the VideoWriter object
    video.release()

def main():
    parser = argparse.ArgumentParser(description='Create a video from PNG images.')
    parser.add_argument('--input_folder', required=True, help='Path to the input folder containing PNG images.')
    parser.add_argument('--output_file', required=True, help='Path to the output video file.')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second for the video.')
    parser.add_argument('--width', type=int, default=512, help='Target width for resizing images.')
    parser.add_argument('--height', type=int, default=256, help='Target height for resizing images.')

    args = parser.parse_args()

    create_video(args.input_folder, args.output_file, args.fps, args.width, args.height)

if __name__ == "__main__":
    main()

