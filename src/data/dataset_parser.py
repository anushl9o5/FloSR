import os
import itertools
import argparse
import numpy as np
import json
from tqdm import tqdm
from pytransform3d.transform_manager import TransformManager
from utility import  load_pickle, decompose_affine_mat

parser = argparse.ArgumentParser()

parser.add_argument('--seqs_root', required=True,
                    help='dir path containing all the captured sequence')
parser.add_argument('--save_dir', required=True,
                    help='dir path to store all the parsed data as json files')
parser.add_argument('--calib_dir_names', nargs='*', 
                     default=['calibration_matrix'],
                     help='possible choice of names for calibration directory')
parser.add_argument('--exts', nargs='*', 
                     default=['pickle'],
                     help='possible extensions for camera parameter file')
parser.add_argument('--cam_extr_fnames', nargs='*', 
                     default=['camera_extrinsic','extrinsic'],
                     help='possible names for camera extrinsics file')
parser.add_argument('--cam_intr_fnames', nargs='*', 
                     default=['camera_intrinsic','intrinsic'],
                     help='possible names for camera intrinsics file')
parser.add_argument('--cam_names', nargs='*', 
                     default=['ar0820','fsc231'],
                     help='possible names for camera intrinsics file')

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# Get list of capture sequences in the sequence root directory
cap_seq_names = os.listdir(args.seqs_root)

for cap_name in tqdm(cap_seq_names):

    try:
        working_calib_dir = None
        working_extr_file = None
        working_intr_file = None

        # check name of calib directory
        for calb_dir in args.calib_dir_names:
            if os.path.isdir(os.path.join(args.seqs_root, cap_name, calb_dir)):
                working_calib_dir = calb_dir
                break
        if working_calib_dir != None:
            # check extrinsics file
            for fname, ext in itertools.product(args.cam_extr_fnames, args.exts):
                if os.path.isfile(os.path.join(args.seqs_root, cap_name, working_calib_dir, fname+'.'+ext)):
                    working_extr_file = fname + '.' + ext
                    break
            # check intrinsics file
            for fname, ext in itertools.product(args.cam_intr_fnames, args.exts):
                if os.path.isfile(os.path.join(args.seqs_root, cap_name, working_calib_dir, fname+'.'+ext)):
                    working_intr_file = fname + '.' + ext
                    break

        if working_extr_file != None and working_intr_file != None:

            data_dict = {}
            data_dict[cap_name] = {}

            intr_data = load_pickle(os.path.join(
                args.seqs_root, cap_name, working_calib_dir, working_intr_file))
            extr_data = load_pickle(os.path.join(
                args.seqs_root, cap_name, working_calib_dir, working_extr_file))

            tm = TransformManager()
            for cam_pair in list(extr_data.keys()):

                cam1, cam2 = tuple([f'{cam}_{num}' for cam, num in zip(
                    cam_pair.split('_')[::2], cam_pair.split('_')[1::2])])

                # extract transformation
                T = np.eye(4)
                R, t = extr_data[cam_pair]['rvec'].reshape(
                    3, 3), extr_data[cam_pair]['tvec'].reshape(3,)
                T[0:3, 0:3] = R

                if np.linalg.norm(t) > 10:
                    T[0:3, 3] = t/1000
                else:
                    T[0:3, 3] = t

                # Add transformation to pytransform3d
                tm.add_transform(cam1, cam2, T)

            cam_types = set(map(lambda x: x.split('_')[0], intr_data.keys()))

            # selecting only the chosen cameras
            cam_types = set(cam_types).intersection(set(args.cam_names))

            for cam_type in cam_types:
                data_dict[cap_name][cam_type] = {
                    'intrinsics': {}, 'extrinsics': {}}

            cam_data = {cam_type: [tag.split('_')[1] for tag in intr_data.keys() if cam_type in tag] for cam_type in cam_types}

            for cam_type in cam_types:
                cam_nums = cam_data[cam_type]
                for cam_num in cam_nums:
                    # intrinsics data
                    data_dict[cap_name][cam_type]['intrinsics'][f'{cam_num}'] = {'K': list(intr_data[f'{cam_type}_{cam_num}']['cam_mat'].astype(float).reshape(-1)),
                                                                                 'dist': list(intr_data[f'{cam_type}_{cam_num}']['dist'].astype(float).reshape(-1))}
                # extriniscs data
                cam_pairs = [f'{a}_{b}' for a, b in itertools.combinations(
                    cam_data[cam_type], r=2)]
                # print(cam_pairs)
                for cam_pair in cam_pairs:
                    R, tvec = decompose_affine_mat(tm.get_transform(f"{cam_type}_{cam_pair.split('_')[0]}",
                                                                    f"{cam_type}_{cam_pair.split('_')[1]}"))

                    data_dict[cap_name][cam_type]['extrinsics'][cam_pair] = {
                        'rmat': list(R.reshape(-1)),
                        'tvec': list(tvec.reshape(-1))
                    }

            with open(os.path.join(args.save_dir, f'{cap_name}.json'), 'w') as fh:
                json.dump(data_dict, fh, sort_keys=True, indent=4)

        else:
            print('Skipping sequence: ' + cap_name)
            continue
    
    except:
        print('Skipping sequence: '+ cap_name)

