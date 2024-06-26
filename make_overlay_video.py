import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm

def overlay_images(input_folder1, input_folder2, output_file, fps=30, target_width=1024, target_height=512, overlay=False):
    # Get the list of PNG files in the first input folder
    image_files1 = [f for f in os.listdir(input_folder1) if f.endswith('.png')]

    # Sort the image files based on their names
    image_files1.sort()

    # Get the list of PNG files in the second input folder
    image_files2 = [f for f in os.listdir(input_folder2) if f.endswith('.png')]

    # Sort the image files based on their names
    image_files2.sort()

    # Ensure that both input folders have the same number of images
    if len(image_files1) != len(image_files2):
        raise ValueError("Input folders must have the same number of images.")

    # Read the dimensions of the first image to set video dimensions
    first_image = cv2.imread(os.path.join(input_folder1, image_files1[0]))
    height, width, layers = first_image.shape

    if overlay:
        # Resize dimensions
        resize_dimensions = (target_width, target_height)
    else:
        resize_dimensions = (target_width*2, target_height)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, resize_dimensions)

    # Iterate through the sorted image files, overlay images, resize them, and add to the video
    for image_file1, image_file2 in tqdm(zip(sorted(image_files1), sorted(image_files2))):
        image_path1 = os.path.join(input_folder1, image_file1)
        image_path2 = os.path.join(input_folder2, image_file2)

        frame1 = cv2.imread(image_path1)
        frame2 = cv2.imread(image_path2)

        if overlay:
            # Overlay images with the same name
            overlay_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
        else:
            overlay_frame = np.hstack([frame1, frame2])


        # Resize the overlayed frame
        resized_frame = cv2.resize(overlay_frame, resize_dimensions)

        video.write(resized_frame)

    # Release the VideoWriter object
    video.release()

def main():
    parser = argparse.ArgumentParser(description='Overlay images from two folders and create a video.')
    parser.add_argument('--input_folder1', required=True, help='Path to the first input folder containing PNG images.')
    parser.add_argument('--input_folder2', required=True, help='Path to the second input folder containing PNG images.')
    parser.add_argument('--output_file', required=True, help='Path to the output video file.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the video.')
    parser.add_argument('--width', type=int, default=1024, help='Target width for resizing images.')
    parser.add_argument('--height', type=int, default=512, help='Target height for resizing images.')
    parser.add_argument('--overlay', type=bool, default=False, help='Flag for making overlays.')

    args = parser.parse_args()

    overlay_images(args.input_folder1, args.input_folder2, args.output_file, args.fps, args.width, args.height, args.overlay)

if __name__ == "__main__":
    main()
