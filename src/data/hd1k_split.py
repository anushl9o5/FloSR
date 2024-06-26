import os
import csv
import glob
import pandas as pd
import numpy as np

from tqdm import tqdm

def parse_data(data_dir, sample_freq=10, flip_aug=True):
    data_dict = {}

    data_split = data_dir.split('/')
    intrinsics_dir = os.path.join(data_dir, "calib")
    img0s = sorted(glob.glob(os.path.join(data_dir, "image_2/*.png")))
    img1s = sorted(glob.glob(os.path.join(data_dir, "image_3/*.png")))
    recording_id = data_split[-1]

    for img0, img1 in zip(img0s[::sample_freq], img1s[::sample_freq]):
        filename = os.path.basename(img0).split('.')[0]
        data_dict["img0"] = data_dict.get("img0", []) + [img0]
        data_dict["img1"] = data_dict.get("img1", []) + [img1]
        intrinsics_file = os.path.join(intrinsics_dir, filename+'.txt')
        data_dict["intrinsics"] = data_dict.get("intrinsics", []) + [intrinsics_file]
        data_dict["id"] = data_dict.get("id", []) + [recording_id]
        data_dict["is_flipped"] = data_dict.get("is_flipped", []) + [False]

        if np.random.rand() > 0.5 and flip_aug:
            filename = os.path.basename(img0).split('.')[0]
            data_dict["img0"] = data_dict.get("img0", []) + [img0]
            data_dict["img1"] = data_dict.get("img1", []) + [img1]
            intrinsics_file = os.path.join(intrinsics_dir, filename+'.txt')
            data_dict["intrinsics"] = data_dict.get("intrinsics", []) + [intrinsics_file]
            data_dict["id"] = data_dict.get("id", []) + [recording_id]
            data_dict["is_flipped"] = data_dict.get("is_flipped", []) + [True]

    return data_dict

def save_splits(data_dict, mode='train'):
    df = pd.DataFrame.from_dict(data_dict)
    print(mode, df['id'].value_counts())
    df.to_csv("{}_dataset.csv".format(mode), index=False, header=False)


if __name__ == "__main__":
    train_dirs = "/nas/EOS/users/anush/HD1K/train"
    test_dirs = "/nas/EOS/users/anush/HD1K/test"

    train_dict = parse_data(train_dirs, sample_freq=1)
    test_dict = parse_data(test_dirs, sample_freq=1, flip_aug=False)

    save_splits(train_dict, mode='train')
    save_splits(test_dict, mode='test')


    