import os
import argparse
import glob
import csv
import cv2
import random

from conv2rectRGB import generate_rgb_image, makedirs
from joblib import Parallel, delayed, cpu_count
from tqdm.contrib import tzip

parser = argparse.ArgumentParser(description='convert RAW frames to RGB and save')
parser.add_argument('--frame_data_dir', required=True,
                     help='path to directory where frame_sample.py results are stored')
parser.add_argument('--result_dir', required=True,
                        help ='directory to store results')
parser.add_argument('--workers', default=10, 
                    type=int,  help ='number of workers (default is 10)')
parser.add_argument('--sample_num', default=-1, 
                    type=int,  help ='number of frames to sample, -1 means use all frames')
parser.add_argument('--start', default=0, 
                    type=int,  help ='start index of all the frames to be converted to RGB')
parser.add_argument('--end', default=-1, 
                    type=int,  help ='end index of all the frames to be converted to RGB')
args = parser.parse_args()


def write_rgb_image(left_raw_path, right_raw_path, camtype, left_save_dir, right_save_dir, extension = 'png'):
    """This function reads left and right raw images and convert them 
       into rgb images and save them.

    Args:
        left_raw_path ([str]): left raw image path
        right_raw_path ([str]): right raw image path
        
    Returns:
        None
    """
    makedirs(left_save_dir)
    makedirs(right_save_dir)

    savenmame_left = os.path.basename(left_raw_path).split('.')[0] + "." + extension
    savenmame_right = os.path.basename(right_raw_path).split('.')[0] + "." + extension

    if os.path.exists(os.path.join(left_save_dir,savenmame_left)) and os.path.exists(os.path.join(right_save_dir,savenmame_right)):
        return
    
    left_rgb, right_rgb = generate_rgb_image(left_raw_path, right_raw_path, camtype, False, False)

    # write images
    cv2.imwrite(os.path.join(left_save_dir,savenmame_left),left_rgb)
    cv2.imwrite(os.path.join(right_save_dir,savenmame_right),right_rgb)

# Read all csv files in sampling frames directory
csv_fpaths = sorted(glob.glob(os.path.join(args.frame_data_dir,"**", "*.csv"),recursive=True))

# Store left, right paths, savepaths and cam types in list 
left_paths = []
right_paths = []
cam_types = []
left_save_dirs = []
right_save_dirs = []

for csv_fpath in csv_fpaths:
    with open(csv_fpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            root_dir, capture, seq, cam_type, left_fid, right_fid = row
            dataset = 'torc' if 'torc' in root_dir else 'continental'   
            left_paths.append(os.path.join(root_dir, capture, seq, cam_type, left_fid+'.raw'))
            right_paths.append(os.path.join(root_dir, capture, seq, cam_type, right_fid+'.raw'))
            cam_types.append(cam_type)
            left_save_dirs.append(os.path.join(args.result_dir, dataset, capture, seq, cam_type, left_fid.split('/')[0]))
            right_save_dirs.append(os.path.join(args.result_dir, dataset, capture, seq, cam_type, right_fid.split('/')[0]))

if args.sample_num > 0:
    sample_indxs = random.sample(range(0,len(left_paths)), args.sample_num)
else:
    sample_indxs = range(args.start, args.end if args.end != -1 else len(left_paths))

left_paths = [left_paths[i] for i in sample_indxs]
right_paths = [right_paths[i] for i in sample_indxs]
cam_types = [cam_types[i] for i in sample_indxs]
left_save_dirs = [left_save_dirs[i] for i in sample_indxs]
right_save_dirs = [right_save_dirs[i] for i in sample_indxs]

# Run the conversion
w = min(cpu_count(), args.workers)
Parallel(n_jobs=w, verbose=10)(delayed(write_rgb_image)(left_path, right_path, cam_type, left_save_dir, right_save_dir) 
                                                for left_path, right_path, cam_type, left_save_dir, right_save_dir in tzip(left_paths, right_paths, cam_types, left_save_dirs, right_save_dirs))


