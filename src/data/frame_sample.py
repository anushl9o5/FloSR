from email import parser
import os
import random
import json
import glob
import pickle
import csv
import argparse
from tqdm import tqdm

from sklearn import datasets

parser = argparse.ArgumentParser(description='construct CSV files for each sequence')
parser.add_argument('--json_root_dir', required=True,
                    help='root to all parsed JSON data from dataset_parser.py')
parser.add_argument('--save_dir', required=True,
                    help='directory to save all the sampled frames')
parser.add_argument('--frames_per_sequence', type = int, default=10000,
                    help='number of frames to be sampled for each sequence')
parser.add_argument('--filter_words', nargs='*', default=['night', 'lidar', 'calib', 'alignment'],
                    help='using few filter words to get rid pof lidar,calibration and night data getting included in the dataset')

parser.add_argument('--cam_pair_filter', nargs='*', default=['0', '3'],
                    help='Camera pairs to filter')

args = parser.parse_args()


root_table = {
    "torc": {
                "wk1":  "/nas/EOS/dataset/torc/wk1/fixed_structure",
                "wk2":  "/nas/EOS/dataset/torc/wk2/fixed_structure",
                "wk3":  "/nas/EOS/dataset/torc/wk3/fixed_structure"
    }
}
header = ['name', 'area', 'country_code2', 'country_code3']

# Get all the parsed sequences from JSON files
json_paths = glob.glob(os.path.join(args.json_root_dir, '**', '*.json'), recursive=True)

counter = 0
for fpath in tqdm(json_paths):

    if 'torc' in fpath:
        dataset = 'torc'
        week, cap_fname = fpath.split('/')[-2:]
    else:
        dataset = 'continental'
        week, cap_fname = None, fpath.split('/')[-1]

    # Get sequence name from file name
    capture = cap_fname.split('.')[0]
    if capture  not in ['20210929', 'CAPT-192', 'CAPT-194', 'CAPT-195', 'CAPT-198', 'CAPT-199', 'CAPT-201', 'CAPT-202', 
                        'CAPT-205', 'CAPT-264', 'CAPT-265', 'CAPT-268', 'CAPT-269', 'CAPT-272', 'CAPT-273', 'CAPT-321', 
                        'CAPT-349', 'CAPT-353', 'CAPT-356', 'CAPT-357','CAPT-411']:
        continue

    # make save directory
    save_dir = os.path.join(args.save_dir, dataset, f"{week if week is not None else '' }", capture)
    os.makedirs(save_dir, exist_ok=True)

    # Load json data
    with open(fpath,'r') as fp:
        cam_data = json.load(fp)

    # load which type of cameras are used in this dataset
    cam_types = list(cam_data[capture].keys())
    cam_nums = {cam_type : list(cam_data[capture][cam_type]['intrinsics'].keys()) for cam_type in cam_types}

    # Load all possible image camera pairs for extrinsics in JSON file
    cam_pairs = {cam_type : list(cam_data[capture][cam_type]['extrinsics'].keys()) for cam_type in cam_types}
    

    # directory to read stored frame
    data_root = root_table[dataset] if week is None else root_table[dataset][week]

    frames_data = []
    for cam_type in cam_types:

        # possible image sequences containing camera name and image#
        seq_paths = glob.glob(f'{data_root}/{capture}/*/{cam_type}/image{cam_nums[cam_type][0]}')

        # filter based on few words
        seq_paths = list(filter(lambda x : all([word not in x for word in args.filter_words]), seq_paths))

        # get sequence names
        seq_names = list(map(lambda x : x.split('/')[-3], seq_paths))
        

        for seq in seq_names:
            for cam1, cam2 in map(lambda x : tuple(x.split('_')), list(cam_pairs.values())[0]):
                if cam1 != str(args.cam_pair_filter[0]) or cam2 != str(args.cam_pair_filter[1]):
                    continue

                try:
                    left_fnames = sorted(os.listdir(os.path.join(data_root, capture, seq, cam_type, 'image'+str(cam1))))
                    right_fnames = sorted(os.listdir(os.path.join(data_root, capture, seq, cam_type, 'image'+str(cam2))))

                    # raising an exception because if the number of files are not same,
                    # two file at same place in sorted order might not be stereo pair
                    assert len(left_fnames) == len(right_fnames)

                    num_images = len(left_fnames)
                    #sample_size = args.frames_per_sequence//len(seq_names) + 1
                    start_id = random.sample(range(0, int(0.8*num_images), 10), 1)[0]
                    new_num_images = len(left_fnames[start_id:])
                    sample_indxs = range(start_id, min(new_num_images, args.frames_per_sequence), 10) #random.sample(range(0, num_images), sample_size)

                    print(capture, len(sample_indxs))
                    for sample_indx in sample_indxs:
                        seq_path = os.path.join('torc', capture, seq, cam_type)
                        left_img_path = os.path.join(seq_path, f"image{cam1}/{left_fnames[sample_indx].split('.')[0]}.png")
                        right_img_path = os.path.join(seq_path, f"image{cam2}/{right_fnames[sample_indx].split('.')[0]}.png")

                        
                        frames_data.append([data_root,capture, seq, cam_type,
                                            f"image{cam1}/{left_fnames[sample_indx].split('.')[0]}",
                                            f"image{cam2}/{right_fnames[sample_indx].split('.')[0]}"])
                        
                        '''
                        frames_data.append(['torc',data_root.split('/')[-2], capture, cam_type, cam1, cam2,
                                            left_img_path,
                                            right_img_path])
                        '''

                        #import pdb; pdb.set_trace();

                except AssertionError:
                    print(f"Number of left({len(left_fnames)}) and right images({len(right_fnames)}) not same!, Skipping  sequence : {seq} image pair : {cam1}, {cam2}")
    
    # save the data as capture file
    if len(frames_data) > 0:
        with open(os.path.join(save_dir, capture + '.csv'), 'w') as fp:
            for row in frames_data:
                csv_writer = csv.writer(fp, delimiter=',')
                csv_writer.writerow(row)
                counter += 1

print("Total number of frames in dataset = " + str(counter))
                    



    