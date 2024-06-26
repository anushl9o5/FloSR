import os
import csv
import glob
import pandas as pd

from tqdm import tqdm

def parse_data(data_dirs, sample_freq=10):
    data_dict = {}

    for data_dir in tqdm(data_dirs):
        data_split = data_dir.split('/')
        extrinsics_file = os.path.join(data_dir, "calibration/egovehicle_SE3_sensor.feather")
        intrinsics_file = os.path.join(data_dir, "calibration/intrinsics.feather")
        img0s = sorted(glob.glob(os.path.join(data_dir, "sensors/cameras/stereo_front_left/*.jpg")))
        img1s = sorted(glob.glob(os.path.join(data_dir, "sensors/cameras/stereo_front_right/*.jpg")))
        recording_id = data_split[-2]
        check_extrinsics = pd.read_feather(extrinsics_file)

        #Argoverse2 has missing extrinsics calibration for stereo left and right, dont' know why? So skipping those
        if len(check_extrinsics) < 11:
            continue

        for img0, img1 in zip(img0s[::sample_freq], img1s[::sample_freq]):
            data_dict["img0"] = data_dict.get("img0", []) + [img0]
            data_dict["img1"] = data_dict.get("img1", []) + [img1]
            data_dict["extrinsics"] = data_dict.get("extrinsics", []) + [extrinsics_file]
            data_dict["intrinsics"] = data_dict.get("intrinsics", []) + [intrinsics_file]
            data_dict["id"] = data_dict.get("id", []) + [recording_id]

    return data_dict

def save_splits(data_dict, mode='train'):
    df = pd.DataFrame.from_dict(data_dict)
    print(mode, df['id'].value_counts())
    df.to_csv("{}_dataset.csv".format(mode), index=False, header=False)


if __name__ == "__main__":
    train_dirs = glob.glob("/nas/EOS/users/anush/argo/train/*/") # Path to the training data
    val_dirs = glob.glob("/nas/EOS/users/anush/argo/val/*/") # Path to the validation data
    test_dirs = glob.glob("/nas/EOS/users/anush/argo/test/*/") # Path to the test data

    train_dict = parse_data(train_dirs, sample_freq=10)
    val_dict = parse_data(val_dirs, sample_freq=10)
    test_dict = parse_data(test_dirs, sample_freq=1)

    save_splits(train_dict, mode='train')
    save_splits(val_dict, mode='val')
    save_splits(test_dict, mode='test')


    